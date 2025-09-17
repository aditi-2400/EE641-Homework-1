import torch
import numpy as np

IMAGE_SIZE = 224
FEATURE_MAP_SIZES = [(56, 56), (28, 28), (14, 14)]
ANCHOR_SCALES = [
    [16, 24, 32],    # for 56x56
    [48, 64, 96],    # for 28x28
    [96, 128, 192],  # for 14x14
]

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.
    
    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size
        
    Returns:
        anchors: List of tensors, each of shape [num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    # For each feature map:
    # 1. Create grid of anchor centers
    # 2. Generate anchors with specified scales and ratios
    # 3. Convert to absolute coordinates
    all_levels = []
    for (H,W), scales in zip(feature_map_sizes,anchor_scales):
        stride_y = image_size/float(H)
        stride_x = image_size/float(W)

        ys = (torch.arange(H, dtype=torch.float32) + 0.5) * stride_y
        xs = (torch.arange(W, dtype=torch.float32) + 0.5) * stride_x
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")        
        centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  

        level_anchors = []
        for s in scales:  
            half = 0.5 * float(s)
            x1y1 = centers - torch.tensor([half, half])
            x2y2 = centers + torch.tensor([half, half])
            level_anchors.append(torch.cat([x1y1, x2y2], dim=1))
        
        level_anchors = torch.cat(level_anchors, dim=0)  
        all_levels.append(level_anchors)

    return all_levels  

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    device = boxes1.device
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    area1 = w1 * h1              
    area2 = w2 * h2              

    # intersection
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  
    wh = (rb - lt).clamp(min=0)                                    
    inter = wh[:, :, 0] * wh[:, :, 1]                               

    # union
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)

def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors
        
    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    device = anchors.device
    A = anchors.shape[0]
    matched_labels = torch.zeros((A,), dtype=torch.long)
    matched_boxes  = torch.zeros((A, 4), dtype=torch.float32)
    pos_mask = torch.zeros((A,), dtype=torch.bool)
    neg_mask = torch.zeros((A,), dtype=torch.bool)
    T = target_boxes.shape[0]
    if T == 0:
        neg_mask[:] = True
        return matched_labels, matched_boxes, pos_mask, neg_mask
    
    target_boxes  = target_boxes.to(device=device, dtype=torch.float32)
    target_labels = target_labels.to(device=device, dtype=torch.long)

    iou = compute_iou(anchors, target_boxes)   
    max_iou, gt_idx = iou.max(dim=1)    
    pos_mask       = torch.zeros_like(max_iou, dtype=torch.bool)     
    neg_mask       = torch.zeros_like(max_iou, dtype=torch.bool)     
    matched_labels = torch.zeros_like(gt_idx,   dtype=torch.long)    
    matched_boxes  = torch.zeros(A, 4, dtype=torch.float32, device=device)
    #print("[match] max_iou device:", max_iou.device, "| pos_mask device:", pos_mask.device)       

    best_anchor_for_gt = iou.argmax(dim=0)     
    pos_mask[best_anchor_for_gt] = True

    # print(
    # "[match] devices:",
    # "anchors", anchors.device,
    # "| tgt_boxes", target_boxes.device if (target_boxes is not None and target_boxes.numel()>0) else None)
    pos_mask = pos_mask | (max_iou >= pos_threshold)
    neg_mask = (max_iou < neg_threshold) & (~pos_mask)

    if pos_mask.any():
        pos_gt_idx = gt_idx[pos_mask]
        matched_boxes[pos_mask]  = target_boxes[pos_gt_idx]
        matched_labels[pos_mask] = target_labels[pos_gt_idx]

    return matched_labels, matched_boxes, pos_mask, neg_mask