import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets


def _xyxy_to_cxcywh(boxes):
    # boxes: [N,4] x1,y1,x2,y2  -> [N,4] cx,cy,w,h
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w  = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    h  = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=1)

def encode_offsets(anchors_xyxy, gt_xyxy):
    """
    SSD/YOLO-style parameterization (no extra scaling here):
      tx = (gx - ax)/aw
      ty = (gy - ay)/ah
      tw = log(gw/aw)
      th = log(gh/ah)
    """
    a = _xyxy_to_cxcywh(anchors_xyxy)
    g = _xyxy_to_cxcywh(gt_xyxy)
    tx = (g[:, 0] - a[:, 0]) / a[:, 2]
    ty = (g[:, 1] - a[:, 1]) / a[:, 3]
    tw = torch.log(g[:, 2] / a[:, 2])
    th = torch.log(g[:, 3] / a[:, 3])
    return torch.stack([tx, ty, tw, th], dim=1)



class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3,
                 w_obj=1.0, w_cls=1.0, w_loc=2.0,
                 pos_thresh=0.5, neg_thresh=0.3, neg_pos_ratio=3):
        super().__init__()
        self.num_classes = num_classes
        self.w_obj = w_obj
        self.w_cls = w_cls
        self.w_loc = w_loc
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.bce = nn.BCEWithLogitsLoss(reduction='none')      
        self.ce  = nn.CrossEntropyLoss(reduction='none')        
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')   
        
    def forward(self, predictions, targets, anchors):
        """
        Compute multi-task loss.
        
        Args:
            predictions: List of tensors from each scale
            targets: List of dicts with 'boxes' and 'labels' for each image
            anchors: List of anchor tensors for each scale
            
        Returns:
            loss_dict: Dict containing:
                - loss_obj: Objectness loss
                - loss_cls: Classification loss  
                - loss_loc: Localization loss
                - loss_total: Weighted sum
        """
        # For each prediction scale:
        # 1. Match anchors to targets
        # 2. Compute objectness loss (BCE)
        # 3. Compute classification loss (CE) for positive anchors
        # 4. Compute localization loss (Smooth L1) for positive anchors
        # 5. Apply hard negative mining (3:1 ratio)
        device = predictions[0].device
        B = predictions[0].shape[0]
        per = 5 + self.num_classes

        total_obj, total_cls, total_loc = 0.0, 0.0, 0.0
        total_pos, total_obj_count = 0, 0

        # iterate scales
        for scale_idx, (pred_s, anchors_s) in enumerate(zip(predictions, anchors)):
            B, Ctot, H, W = pred_s.shape
            A = Ctot // per
            pred_s = pred_s.permute(0, 2, 3, 1).contiguous().view(B, H*W*A, per)

            pred_loc  = pred_s[..., 0:4]              
            pred_obj  = pred_s[..., 4]               
            pred_cls  = pred_s[..., 5:]               
            N = pred_s.shape[1]

            assert anchors_s.shape[0] == N, "anchors count must match predictions per scale"

            # per-image matching & loss accumulation
            for b in range(B):
                t = targets[b]
                gt_boxes  = t["boxes"].to(device)    
                gt_labels = t["labels"].to(device)   

                m_labels, m_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anchors_s.to(device), gt_boxes, gt_labels,
                    pos_threshold=self.pos_thresh, neg_threshold=self.neg_thresh
                )

                obj_tgt = pos_mask.float()            
                # per-anchor BCE loss
                obj_loss_all = self.bce(pred_obj[b], obj_tgt)  

                # hard negative mining: select among negatives
                sel_neg_mask = self.hard_negative_mining(obj_loss_all, pos_mask, neg_mask, ratio=self.neg_pos_ratio)

                # objectness loss: positives + selected negatives
                obj_mask = pos_mask | sel_neg_mask
                obj_loss = obj_loss_all[obj_mask].sum()
                total_obj += obj_loss
                total_obj_count += int(obj_mask.sum().item())

                # Classification loss
                if pos_mask.any():
                    cls_logits_pos = pred_cls[b][pos_mask]         
                    cls_targets_pos = m_labels[pos_mask]            
                    cls_loss = self.ce(cls_logits_pos, cls_targets_pos).sum()
                    total_cls += cls_loss

                    # localization loss (positives only) 
                    loc_targets = encode_offsets(anchors_s[pos_mask].to(device), m_boxes[pos_mask])
                    loc_pred    = pred_loc[b][pos_mask]              
                    loc_loss = self.smoothl1(loc_pred, loc_targets).sum()  
                    total_loc += loc_loss

                    total_pos += int(pos_mask.sum().item())

        denom_pos = max(total_pos, 1)
        denom_obj = max(total_obj_count, 1)

        loss_obj = (total_obj / denom_obj) * self.w_obj
        loss_cls = (total_cls / denom_pos) * self.w_cls
        loss_loc = (total_loc / denom_pos) * self.w_loc
        loss_total = loss_obj + loss_cls + loss_loc

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_tot": loss_total
        }
    
    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.
        
        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio
            
        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """
        num_pos = int(pos_mask.sum().item())
        k = ratio * num_pos if num_pos > 0 else min(int(neg_mask.sum().item()), 100)

        if k == 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)

        # losses for negative anchors only
        neg_losses = loss.clone()
        neg_losses[~neg_mask] = -1.0  

        # top-k by loss
        k = min(k, int(neg_mask.sum().item()))
        if k <= 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)

        topk_vals, topk_idx = torch.topk(neg_losses, k=k, largest=True, sorted=False)
        selected = torch.zeros_like(neg_mask, dtype=torch.bool)
        selected[topk_idx] = True
        # return only those that are genuine negatives
        return selected & neg_mask