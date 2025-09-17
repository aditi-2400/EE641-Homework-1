import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import torch.nn.functional as F
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "datasets" / "keypoints"   

TRAIN_IMG_DIR = DATA_DIR / "train"
TRAIN_ANN = DATA_DIR / "train_annotations.json"
VAL_IMG_DIR = DATA_DIR / "val"
VAL_ANN = DATA_DIR / "val_annotations.json"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = RESULTS_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_WEIGHTS = RESULTS_DIR / "heatmap_model.pth"
REGRESSION_WEIGHTS = RESULTS_DIR / "regression_model.pth"
LOG_PATH = RESULTS_DIR / "training_log.json"

IMAGE_SIZE   = 128
HEATMAP_SIZE = 64
NUM_KPS      = 5

PCK_THRESHOLDS = np.linspace(0.01, 0.20, 10)

def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def extract_keypoints_from_heatmaps(heatmaps, up_factor=2):
    """
    Extract (x, y) coordinates from heatmaps.
    
    Args:
        heatmaps: Tensor of shape [batch, num_keypoints, H, W]
        
    Returns:
        coords: Tensor of shape [batch, num_keypoints, 2]
    """
    # Find argmax location in each heatmap
    # Convert to (x, y) coordinates
    B, K, Hh, Wh = heatmaps.shape
    p = F.softmax(heatmaps.view(B * K, -1), dim=-1).view(B * K, Hh, Wh)

    ys = torch.linspace(0, Hh - 1, Hh, device=heatmaps.device)
    xs = torch.linspace(0, Wh - 1, Wh, device=heatmaps.device)
    grid_y = ys.view(1, Hh, 1).expand(B * K, Hh, Wh)
    grid_x = xs.view(1, 1, Wh).expand(B * K, Hh, Wh)

    ey = (p * grid_y).view(B, K, -1).sum(-1)  
    ex = (p * grid_x).view(B, K, -1).sum(-1)
    coords_h = torch.stack([ex, ey], dim=-1)  
    coords_px = coords_h * float(up_factor)   
    return coords_px  

def bbox_diagonal(gt_xy: torch.Tensor) -> torch.Tensor:
    """
    gt_xy: [B, K, 2] in pixels
    Returns: [B] diagonal lengths of the tight GT box
    """
    x_min, _ = gt_xy[:,:, 0].min(dim=1)
    y_min, _ = gt_xy[:,:, 1].min(dim=1)
    x_max, _ = gt_xy[:,:, 0].max(dim=1)
    y_max, _ = gt_xy[:,:, 1].max(dim=1)
    w = (x_max - x_min).clamp(min=1e-6)
    h = (y_max - y_min).clamp(min=1e-6)
    return torch.sqrt(w * w + h * h)

@torch.no_grad()
def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    """
    Compute PCK at various thresholds.
    
    Args:
        predictions: Tensor of shape [N, num_keypoints, 2]
        ground_truths: Tensor of shape [N, num_keypoints, 2]
        thresholds: List of threshold values (as fraction of normalization)
        normalize_by: 'bbox' for bounding box diagonal, 'torso' for torso length
        
    Returns:
        pck_values: Dict mapping threshold to accuracy
    """
    # For each threshold:
    # Count keypoints within threshold distance of ground truth
    assert predictions.shape == ground_truths.shape
    N, K, _ = predictions.shape

    d = torch.linalg.norm(predictions - ground_truths, dim=-1)  

    if normalize_by == 'bbox':
        norm = bbox_diagonal(ground_truths)                      
    elif normalize_by == 'image':
        norm = torch.full((N,), float(max(IMAGE_SIZE, IMAGE_SIZE)), device=predictions.device)
    else:
        raise ValueError("normalize_by must be 'bbox' or 'image'")

    norm = norm[:, None].expand(N, K)

    pck_vals = {}
    for t in thresholds:
        thr = t * norm  
        acc = (d <= thr).float().mean().item()  
        pck_vals[float(t)] = acc
    return pck_vals

def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    """
    Plot PCK curves comparing both methods.
    """
    ts = sorted(pck_heatmap.keys())
    y_h = [pck_heatmap[t] for t in ts]
    y_r = [pck_regression[t] for t in ts]
    plt.figure(figsize=(6, 4))
    plt.plot(ts, y_h, marker='o', label='Heatmap')
    plt.plot(ts, y_r, marker='s', label='Regression')
    plt.xlabel('Threshold (fraction of normalizer)')
    plt.ylabel('PCK')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_predictions(image_tensor, pred_keypoints, gt_keypoints, save_path, radius=3):
    """
    Visualize predicted and ground truth keypoints on image.
    """
    img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode='L').convert('RGB')
    draw = ImageDraw.Draw(pil)
 
    for x, y in gt_keypoints.cpu().numpy():
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=2)
   
    for x, y in pred_keypoints.cpu().numpy():
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 0, 0), width=2)
    pil.save(save_path)

@torch.no_grad()
def save_heatmap_grid(hm_logits: torch.Tensor, save_path: Path, max_cols: int = 5):
    """
    Save a simple grid of heatmaps (max over K) for a batch.
    hm_logits: [B, K, Hh, Wh]
    """
    B, K, Hh, Wh = hm_logits.shape
    hm = torch.sigmoid(hm_logits).amax(dim=1).cpu().numpy()  # [B, Hh, Wh]
    cols = min(B, max_cols)
    rows = int(np.ceil(B / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis('off')
        if i < B:
            ax.imshow(hm[i], vmin=0, vmax=1, cmap='magma')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

@torch.no_grad()
def run_evaluation(normalize_by='bbox'):
    dev = device()
    print("Device:", dev)

    val_ds = KeypointDataset(str(VAL_IMG_DIR), str(VAL_ANN),
                             output_type="regression",  
                             heatmap_size=HEATMAP_SIZE, sigma=2.0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    hm_model = HeatmapNet(num_keypoints=NUM_KPS).to(dev)
    rg_model = RegressionNet(num_keypoints=NUM_KPS).to(dev)

    if HEATMAP_WEIGHTS.exists():
        hm_model.load_state_dict(torch.load(HEATMAP_WEIGHTS, map_location=dev))
        print("Loaded:", HEATMAP_WEIGHTS)
    else:
        print("WARN: heatmap weights not found:", HEATMAP_WEIGHTS)

    if REGRESSION_WEIGHTS.exists():
        rg_model.load_state_dict(torch.load(REGRESSION_WEIGHTS, map_location=dev))
        print("Loaded:", REGRESSION_WEIGHTS)
    else:
        print("WARN: regression weights not found:", REGRESSION_WEIGHTS)

    hm_model.eval()
    rg_model.eval()

    all_imgs  = []
    all_gt_xy = []
    all_hm_xy = []
    all_rg_xy = []

    for imgs, tgt_reg in val_loader:
        B = imgs.size(0)
        imgs = imgs.to(dev).float()

        gt_xy = (tgt_reg.view(B, NUM_KPS, 2) * float(IMAGE_SIZE)).to(dev)

        hm_logits = hm_model(imgs)                            
        hm_xy     = extract_keypoints_from_heatmaps(hm_logits, up_factor=IMAGE_SIZE // HEATMAP_SIZE)

        rg_out = rg_model(imgs)                          
        rg_xy  = (rg_out.view(B, NUM_KPS, 2) * float(IMAGE_SIZE))

        all_imgs.append(imgs.cpu())
        all_gt_xy.append(gt_xy.cpu())
        all_hm_xy.append(hm_xy.cpu())
        all_rg_xy.append(rg_xy.cpu())

        if len(all_imgs) == 1:
            save_heatmap_grid(hm_logits, VIS_DIR / "val_heatmaps_grid.png")

    imgs_t  = torch.cat(all_imgs,  dim=0)  
    gt_xy_t = torch.cat(all_gt_xy, dim=0)  
    hm_xy_t = torch.cat(all_hm_xy, dim=0)  
    rg_xy_t = torch.cat(all_rg_xy, dim=0) 

    pck_h = compute_pck(hm_xy_t.to(dev), gt_xy_t.to(dev), PCK_THRESHOLDS, normalize_by=normalize_by)
    pck_r = compute_pck(rg_xy_t.to(dev), gt_xy_t.to(dev), PCK_THRESHOLDS, normalize_by=normalize_by)

    # Plot PCK
    plot_pck_curves(pck_h, pck_r, VIS_DIR / f"pck_{normalize_by}.png")

    with open(RESULTS_DIR / f"pck_{normalize_by}.json", "w") as f:
        json.dump({"normalize_by": normalize_by, "thresholds": list(map(float, PCK_THRESHOLDS)),
                   "heatmap": pck_h, "regression": pck_r}, f, indent=2)


    with torch.no_grad():
        
        err_h = torch.linalg.norm(hm_xy_t - gt_xy_t, dim=-1).mean(dim=-1)  
        err_r = torch.linalg.norm(rg_xy_t - gt_xy_t, dim=-1).mean(dim=-1)
        N = imgs_t.size(0)

        best_idx  = torch.topk(-err_h, k=min(4, N)).indices.tolist()
        worst_idx = torch.topk(err_h,  k=min(4, N)).indices.tolist()
        for i, idx in enumerate(best_idx):
            visualize_predictions(imgs_t[idx], hm_xy_t[idx], gt_xy_t[idx], VIS_DIR / f"hm_best_{i}.png")
        for i, idx in enumerate(worst_idx):
            visualize_predictions(imgs_t[idx], hm_xy_t[idx], gt_xy_t[idx], VIS_DIR / f"hm_worst_{i}.png")

        for i in range(min(4, N)):
            visualize_predictions(imgs_t[i], rg_xy_t[i], gt_xy_t[i], VIS_DIR / f"rg_{i}.png")

    print("Saved:",
          VIS_DIR / f"pck_{normalize_by}.png",
          VIS_DIR / "val_heatmaps_grid.png",
          "and sample overlays in", VIS_DIR)

def main():
    run_evaluation(normalize_by='bbox')

if __name__ == "__main__":
    main()