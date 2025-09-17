import os, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from dataset import ShapeDetectionDataset, detection_collate
from model import MultiScaleDetector
from utils import generate_anchors


BASE_DIR = Path(__file__).resolve().parent

TRAIN_IMG_DIR = BASE_DIR.parent / "datasets" / "detection" / "train"
TRAIN_ANN     = BASE_DIR.parent / "datasets" / "detection" / "train_annotations.json"
VAL_IMG_DIR   = BASE_DIR.parent / "datasets" / "detection" / "val"
VAL_ANN       = BASE_DIR.parent / "datasets" / "detection" / "val_annotations.json"

RESULTS_DIR      = BASE_DIR / "results"
BEST_MODEL_PATH  = RESULTS_DIR / "best_model.pth"
LOG_PATH         = RESULTS_DIR / "training_log.json"
VIS_DIR          = RESULTS_DIR / "visualizations"


IMAGE_SIZE   = 224
NUM_CLASSES  = 3
CLASS_NAMES  = ["circle", "square", "triangle"]
NUM_ANCHORS  = 3
OBJ_THRESH   = 0.3
CONF_THRESH  = 0.3
NMS_IOU      = 0.5
AP_IOU       = 0.5
MAX_VIS_IMGS = 10

def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    cx = (xyxy[:,0] + xyxy[:,2]) * 0.5
    cy = (xyxy[:,1] + xyxy[:,3]) * 0.5
    w  = (xyxy[:,2] - xyxy[:,0]).clamp(min=1e-6)
    h  = (xyxy[:,3] - xyxy[:,1]).clamp(min=1e-6)
    return torch.stack([cx,cy,w,h], dim=1)

def cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = cxcywh.unbind(dim=1)
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return torch.stack([x1,y1,x2,y2], dim=1)

def decode_offsets(anchors_xyxy: torch.Tensor, pred_t: torch.Tensor, variances=(0.1,0.1,0.2,0.2)) -> torch.Tensor:
    # inverse of encode: tx=(gx-ax)/aw, ty=(gy-ay)/ah, tw=log(gw/aw), th=log(gh/ah)
    a = xyxy_to_cxcywh(anchors_xyxy)
    v = torch.tensor(variances, device=pred_t.device, dtype=pred_t.dtype)
    t = pred_t * v
    gx = t[:,0]*a[:,2] + a[:,0]
    gy = t[:,1]*a[:,3] + a[:,1]
    gw = a[:,2] * torch.exp(t[:,2])
    gh = a[:,3] * torch.exp(t[:,3])
    return cxcywh_to_xyxy(torch.stack([gx,gy,gw,gh], dim=1))

def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel()==0 or boxes2.numel()==0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0)*(boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0)*(boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    lt = torch.maximum(boxes1[:,None,:2], boxes2[None,:, :2])
    rb = torch.minimum(boxes1[:,None, 2:], boxes2[None,:, 2:])
    wh = (rb-lt).clamp(min=0)
    inter = wh[:,:,0]*wh[:,:,1]
    union = area1[:,None] + area2[None,:] - inter
    return inter/(union + 1e-7)

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    if boxes.numel()==0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = []
    idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0]
        keep.append(i.item())
        if idx.numel()==1: break
        ious = box_iou_matrix(boxes[i].unsqueeze(0), boxes[idx[1:]]).squeeze(0)
        idx = idx[1:][ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

@torch.no_grad()
def forward_and_decode(model, images: torch.Tensor, anchors_per_level):
    """
    Returns per-image list of dicts:
      {"boxes":[K,4], "scores":[K], "labels":[K], "scale_ids":[K]}
    """
    model.eval()
    preds = model(images)  # list of [B, A*(5+C), H, W]
    B = images.size(0)
    per = 5 + NUM_CLASSES

    outputs = []
    for b in range(B):
        all_boxes, all_scores, all_labels, all_scale_ids = [], [], [], []
        for s, (p_s, anchors_s) in enumerate(zip(preds, anchors_per_level)):
            _, Ctot, H, W = p_s.shape
            A = Ctot // per
            logits = p_s.permute(0,2,3,1).contiguous().view(B, H*W*A, per)
            loc = logits[b, :, :4]               # [N,4]
            obj = torch.sigmoid(logits[b, :, 4]) # [N]
            cls = torch.softmax(logits[b, :, 5:], dim=1)  # [N,C]
            conf = obj[:, None] * cls            # [N,C]

            keep_obj = obj >= OBJ_THRESH
            if keep_obj.any():
                dec = decode_offsets(anchors_s[keep_obj].to(images.device), loc[keep_obj])
                conf_m = conf[keep_obj]
                for c in range(NUM_CLASSES):
                    sc = conf_m[:, c]
                    k = sc >= CONF_THRESH
                    if k.any():
                        boxes_c = dec[k]
                        scores_c = sc[k]
                        keep = nms(boxes_c, scores_c, NMS_IOU)
                        if keep.numel() > 0:
                            all_boxes.append(boxes_c[keep]); all_scores.append(scores_c[keep])
                            all_labels.append(torch.full((keep.numel(),), c, dtype=torch.long, device=images.device))
                            all_scale_ids.append(torch.full((keep.numel(),), s, dtype=torch.long, device=images.device))

        if len(all_boxes)==0:
            outputs.append({
                "boxes": torch.zeros((0,4), device=images.device),
                "scores": torch.zeros((0,), device=images.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=images.device),
                "scale_ids": torch.zeros((0,), dtype=torch.long, device=images.device),
            })
        else:
            outputs.append({
                "boxes": torch.cat(all_boxes, 0),
                "scores": torch.cat(all_scores, 0),
                "labels": torch.cat(all_labels, 0),
                "scale_ids": torch.cat(all_scale_ids, 0),
            })
    return outputs



def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """Compute Average Precision for a single class."""
    aps = {}
    device = predictions[0]["boxes"].device if predictions else torch.device("cpu")
    for c in range(NUM_CLASSES):
        pred_list, gt_per_img, gt_used = [], [], {}
        for img_i, (p, g) in enumerate(zip(predictions, ground_truths)):
            if p["boxes"].numel() > 0:
                mask = (p["labels"] == c)
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
                for j in idxs:
                    pred_list.append((img_i, float(p["scores"][j].item()), p["boxes"][j]))
            if g["boxes"].numel() > 0:
                m = (g["labels"] == c)
                boxes_c = g["boxes"][m]
                for gi in range(boxes_c.size(0)): gt_used[(img_i, gi)] = False
            else:
                boxes_c = torch.zeros((0,4), device=device)
            gt_per_img.append({"boxes": boxes_c})

        pred_list.sort(key=lambda x: x[1], reverse=True)

        tp, fp = [], []
        total_gt = sum(gt_per_img[i]["boxes"].size(0) for i in range(len(gt_per_img)))
        for (img_i, score, pbox) in pred_list:
            gboxes = gt_per_img[img_i]["boxes"]
            if gboxes.numel()==0:
                fp.append(1); tp.append(0); continue
            ious = box_iou_matrix(pbox.unsqueeze(0), gboxes).squeeze(0)
            miou, m = ious.max(dim=0)
            m = int(m.item())
            if miou.item() >= iou_threshold and not gt_used[(img_i, m)]:
                tp.append(1); fp.append(0); gt_used[(img_i, m)] = True
            else:
                fp.append(1); tp.append(0)

        if total_gt == 0:
            aps[c] = 0.0; continue

        tp = np.cumsum(np.array(tp, dtype=np.float32))
        fp = np.cumsum(np.array(fp, dtype=np.float32))
        rec = tp / (total_gt + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        for i in range(len(prec)-2, -1, -1):
            prec[i] = max(prec[i], prec[i+1])
        ap, prev_r = 0.0, 0.0
        for p, r in zip(prec, rec):
            ap += p * max(r - prev_r, 0.0); prev_r = r
        aps[c] = float(ap)
    return aps

def visualize_detections(image, predictions, ground_truths, save_path: Path):
    """
    GT: thick green outline + translucent green fill + 'G:<class>' tag
    Preds: colored outline (red/blue/orange) + 'P:<class>:score' tag
    Draw GT and preds on separate overlays so preds are never hidden.
    """
    base = image.convert("RGB")
    W, H = base.size

    gt_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(gt_ov, "RGBA")

    if ground_truths["boxes"].numel() > 0:
        for (x1, y1, x2, y2), lab in zip(
            ground_truths["boxes"].cpu().tolist(),
            ground_truths["labels"].cpu().tolist()
        ):

            gdraw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, 64), outline=(0, 180, 0, 255), width=4)
            gdraw.text((x1 + 2, max(0, y1 - 12)), f"G:{CLASS_NAMES[int(lab)]}", fill=(0, 120, 0, 255))

    out = Image.alpha_composite(base.convert("RGBA"), gt_ov)


    pred_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(pred_ov, "RGBA")

    colors_rgb = [
        (255, 60, 60),    # circle
        (60, 160, 255),   # square
        (255, 170, 0),    # triangle
    ]

    n_pred = int(predictions["boxes"].shape[0]) if "boxes" in predictions else 0
    if n_pred > 0:
        for (x1, y1, x2, y2), score, lab in zip(
            predictions["boxes"].cpu().tolist(),
            predictions["scores"].cpu().tolist(),
            predictions["labels"].cpu().tolist()
        ):
            col = colors_rgb[int(lab) % len(colors_rgb)]
            # draw outline with width=3; text in same RGB color
            pdraw.rectangle([x1, y1, x2, y2], outline=col + (255,), width=3)
            pdraw.text((x1 + 2, max(0, y1 - 12)), f"P:{CLASS_NAMES[int(lab)]}:{score:.2f}", fill=col + (255,))

    out = Image.alpha_composite(out, pred_ov).convert("RGB")


    legend = Image.new("RGB", (150, 50), (255, 255, 255))
    ld = ImageDraw.Draw(legend)
    ld.rectangle([6, 6, 26, 26], fill=(170, 255, 170), outline=(0, 180, 0), width=2)
    ld.text((32, 8), "GT (green)", fill=(0, 0, 0))
    ld.rectangle([6, 28, 26, 46], outline=(255, 60, 60), width=3)
    ld.text((32, 28), "Pred (colored)", fill=(0, 0, 0))
    out.paste(legend, (5, 5))

    if n_pred == 0:
        ImageDraw.Draw(out).text((5, H - 14), "No predictions", fill=(255, 0, 0))

    out.save(save_path)



@torch.no_grad()
def analyze_scale_performance(model, dataloader, anchors_per_level, save_path: Path):
    """Analyze which scales detect which object sizes."""
    # Generate statistics on detection performance per scale
    # Create visualizations showing scale specialization
    device = next(model.parameters()).device
    bins = [0,32,64,128,10_000]; names = ["S","M","L","XL"]
    stats = torch.zeros((len(bins)-1, 3), dtype=torch.long)
    for imgs, targets in dataloader:
        imgs = [im.to(device) for im in imgs]
        dets = forward_and_decode(model, torch.stack(imgs,0), anchors_per_level)
        for i, d in enumerate(dets):
            gtb = targets[i]["boxes"].to(device)
            if gtb.numel()==0 or d["boxes"].numel()==0: continue
            ious = box_iou_matrix(d["boxes"], gtb)
            _, m = ious.max(dim=1)
            mt = gtb[m]
            side = torch.sqrt((mt[:,2]-mt[:,0])*(mt[:,3]-mt[:,1]))
            for sid, s in zip(d["scale_ids"].tolist(), side.tolist()):
                for bi in range(len(bins)-1):
                    if bins[bi] <= s < bins[bi+1]:
                        stats[bi, sid] += 1; break
    # bar chart
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(6,4))
    for s in range(3):
        plt.bar(x+(s-1)*w, stats[:,s].cpu().numpy(), width=w, label=f"Scale{s}")
    plt.xticks(x, names); plt.ylabel("#detections"); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    return stats

def plot_training_curves(log_path,out_dir):
    if not log_path.exists(): return
    with open(log_path) as f: log=json.load(f)
    ep=[r["epoch"] for r in log]
    for key in ["tot","obj","cls","loc"]:
        tr=[r["train"][key] for r in log]; va=[r["val"][key] for r in log]
        plt.figure(); plt.plot(ep,tr,label="train"); plt.plot(ep,va,label="val")
        plt.legend(); plt.title(key); plt.savefig(out_dir/f"loss_{key}.png"); plt.close()

def _gt_side_lengths(boxes: torch.Tensor) -> torch.Tensor:

    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return torch.sqrt(w * h)

@torch.no_grad()
def compute_anchor_coverage_per_scale(anchors_per_level, gts_list):
    """
    For each scale s:
      - For every GT, compute the max IoU over anchors at that scale.
      - Record GT side length too (sqrt area).
    Returns:
      per_scale_sizes: list of 1D np arrays (GT sizes) per scale
      per_scale_iou:   list of 1D np arrays (max IoU)   per scale
    """
    S = len(anchors_per_level)
    per_scale_sizes = [[] for _ in range(S)]
    per_scale_iou   = [[] for _ in range(S)]

    for gt in gts_list:
        boxes = gt["boxes"]
        if boxes.numel() == 0:
            continue
        sizes = _gt_side_lengths(boxes)  
        for s, A in enumerate(anchors_per_level):
            iou = box_iou_matrix(A.to(boxes.device), boxes) 
            max_iou_per_gt, _ = iou.max(dim=0)               
            per_scale_sizes[s].extend(sizes.cpu().tolist())
            per_scale_iou[s].extend(max_iou_per_gt.cpu().tolist())

    per_scale_sizes = [np.asarray(x, dtype=np.float32) for x in per_scale_sizes]
    per_scale_iou   = [np.asarray(x, dtype=np.float32) for x in per_scale_iou]
    return per_scale_sizes, per_scale_iou

def plot_anchor_hist_per_scale(anchors_per_level, per_scale_sizes, save_dir):
    """Histogram of GT sizes per scale + vertical lines at anchor side lengths."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for s, (A, sizes) in enumerate(zip(anchors_per_level, per_scale_sizes)):
        plt.figure(figsize=(6,4))
        if sizes.size > 0:
            plt.hist(sizes, bins=20, alpha=0.7, label="GT side length")

        w = (A[:, 2] - A[:, 0]).unique().cpu().numpy()
        w = np.sort(w)
        for a in w:
            plt.axvline(a, ls="--", alpha=0.5)
        plt.title(f"Anchor coverage — scale {s}")
        plt.xlabel("Size (pixels)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"anchor_coverage_scale{s}.png", dpi=150)
        plt.close()

def plot_iou_by_size(per_scale_sizes, per_scale_iou, image_size, save_path):
    """Mean max-IoU vs GT size curve for each scale."""
    plt.figure(figsize=(7,4))
    bins = np.linspace(0, image_size, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for s, (sizes, ious) in enumerate(zip(per_scale_sizes, per_scale_iou)):
        if sizes.size == 0:
            continue
        mean_iou = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (sizes >= b0) & (sizes < b1)
            mean_iou.append(ious[m].mean() if m.any() else np.nan)
        plt.plot(centers, mean_iou, marker="o", label=f"scale {s}")
    plt.ylim(0, 1)
    plt.xlabel("GT size (pixels)")
    plt.ylabel("Mean max IoU to anchors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_fraction_matched(per_scale_iou, thr, save_path):
    """Fraction of GT matched (max IoU ≥ thr) per scale."""
    frac = []
    for ious in per_scale_iou:
        frac.append(float((ious >= thr).mean()) if ious.size > 0 else 0.0)
    x = np.arange(len(frac))
    plt.figure(figsize=(5,4))
    plt.bar(x, frac, width=0.6)
    plt.xticks(x, [f"scale {i}" for i in x])
    plt.ylim(0, 1)
    plt.ylabel(f"Fraction GT with max IoU ≥ {thr}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def anchor_coverage_visualizations(anchors_per_level, all_gts, image_size, out_dir):
    """
    Convenience wrapper to generate all anchor-coverage plots.
    Saves:
      - anchor_coverage_scale*.png   (per-scale hist)
      - anchor_coverage_iou_by_size.png
      - anchor_coverage_fraction_05.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scale_sizes, per_scale_iou = compute_anchor_coverage_per_scale(anchors_per_level, all_gts)
    plot_anchor_hist_per_scale(anchors_per_level, per_scale_sizes, out_dir)
    plot_iou_by_size(per_scale_sizes, per_scale_iou, image_size, out_dir / "anchor_coverage_iou_by_size.png")
    plot_fraction_matched(per_scale_iou, thr=0.5, save_path=out_dir / "anchor_coverage_fraction_05.png")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    # val data
    val_ds = ShapeDetectionDataset(str(VAL_IMG_DIR), str(VAL_ANN))
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=detection_collate)

    # model
    model = MultiScaleDetector(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    if BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("Loaded:", BEST_MODEL_PATH)
    model.eval()


    with torch.no_grad():
        dummy = torch.zeros(1,3,IMAGE_SIZE,IMAGE_SIZE, device=device)
        pred_shapes = [(p.shape[2], p.shape[3]) for p in model(dummy)]
    print("Feature map sizes:", pred_shapes)

    ANCHOR_SCALES = [[16,24,32],[48,64,96],[96,128,192]]
    anchors_per_level = generate_anchors(pred_shapes, ANCHOR_SCALES, image_size=IMAGE_SIZE)
    anchors_per_level = [a.to(device) for a in anchors_per_level]

    # run inference and collect preds/gts
    all_preds, all_gts, gt_sizes = [], [], []
    vis_saved = 0

    for imgs, targets in val_loader:
        imgs_d = [im.to(device) for im in imgs]
        dets = forward_and_decode(model, torch.stack(imgs_d,0), anchors_per_level)
        for i in range(len(imgs)):
            all_preds.append({k: dets[i][k] for k in ["boxes","scores","labels"]})
            gtb = targets[i]["boxes"].to(device); gtl = targets[i]["labels"].to(device)
            all_gts.append({"boxes": gtb, "labels": gtl})
            if gtb.numel() > 0:
                sides = torch.sqrt((gtb[:,2]-gtb[:,0])*(gtb[:,3]-gtb[:,1]))
                gt_sizes += sides.cpu().tolist()
            if vis_saved < MAX_VIS_IMGS:
                img_path = Path(val_ds.image_dir) / val_ds.images[i]["file_name"]
                visualize_detections(Image.open(img_path).convert("RGB"),
                                     dets[i], {"boxes": gtb, "labels": gtl},
                                     VIS_DIR / f"val_det_{vis_saved:02d}.png")
                vis_saved += 1

    aps = compute_ap(all_preds, all_gts, iou_threshold=AP_IOU)
    print("AP@0.5:", {CLASS_NAMES[c]: round(aps[c],4) for c in aps})
    with open(VIS_DIR / "metrics.json", "w") as f:
        json.dump({"AP@0.5": {CLASS_NAMES[c]: float(aps[c]) for c in aps}}, f, indent=2)

    # extra plots
    analyze_scale_performance(model, val_loader, anchors_per_level, VIS_DIR / "scale_specialization.png")
    if LOG_PATH.exists():
        plot_training_curves(LOG_PATH, VIS_DIR)
        print("Saved curves from:", LOG_PATH)

    print("Visualizations written to:", VIS_DIR)

    anchor_coverage_visualizations(
    anchors_per_level=anchors_per_level,
    all_gts=all_gts,
    image_size=IMAGE_SIZE,
    out_dir=VIS_DIR)


if __name__ == "__main__":
    main()