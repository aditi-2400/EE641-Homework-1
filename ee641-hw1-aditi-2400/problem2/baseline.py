from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, visualize_predictions
from train import get_device

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "datasets" / "keypoints"  

TRAIN_IMG_DIR = DATA_DIR / "train"
TRAIN_ANN     = DATA_DIR / "train_annotations.json"
VAL_IMG_DIR   = DATA_DIR / "val"
VAL_ANN       = DATA_DIR / "val_annotations.json" 

TEST_IMG_DIR   = DATA_DIR / "val"
TEST_ANN       = DATA_DIR / "val_annotations.json" 

RESULTS_DIR = BASE_DIR / "results"
VIS_DIR     = RESULTS_DIR / "visualizations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE   = 128
DEFAULT_HM   = 64
NUM_KPS      = 5

@torch.no_grad()
def mean_pixel_error(pred_xy, gt_xy):
    return torch.linalg.norm(pred_xy - gt_xy, dim=-1).mean().item()

@torch.no_grad()
def pck(pred_xy: torch.Tensor, gt_xy: torch.Tensor, size: int = 128, alpha: float = 0.05) -> float:
    thr = alpha * size
    d = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [N,K]
    return (d <= thr).float().mean().item()

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


class HeatmapNetNoSkip(nn.Module):
    """
    A no-skip connections variant
    """
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.backbone = HeatmapNet(num_keypoints=num_keypoints).enc  # reuse Encoder
        # simple up path: 8->16->32->64 with convs
        self.up16 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.up64 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2, 0, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
        )
        self.head = nn.Conv2d(32, num_keypoints, 1)

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)  # ignore f2/f3
        x = self.up16(f4)
        x = self.up32(x)
        x = self.up64(x)
        return self.head(x)
    

def train_heatmap_once(model, train_loader, val_loader, device, epochs=6, lr=1e-3,
                       target_hm_size=64):
    """
    Train a heatmap model for a few epochs. If target_hm_size != 64, we resize
    predictions or targets so shapes match on the fly (keeps code simple).
    """
    criterion = nn.BCEWithLogitsLoss()
    optim_ = torch.optim.Adam(model.parameters(), lr=lr)

    def _resize_to(pred_logits, target):
        # pred_logits: [B,K,Hp,Wp], target: [B,K,Ht,Wt]
        Hp, Wp = pred_logits.shape[-2:]
        Ht, Wt = target.shape[-2:]
        if (Hp, Wp) == (Ht, Wt):
            return pred_logits, target
        # resize logits to target size (safer than resizing targets)
        pred_resized = F.interpolate(pred_logits, size=(Ht, Wt), mode='bilinear', align_corners=False)
        return pred_resized, target

    best_val = float('inf')
    for ep in range(1, epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for imgs, targs in train_loader:
            imgs  = imgs.to(device).float()
            targs = targs.to(device).float()  # [B,K,Ht,Wt]
            optim_.zero_grad()
            logits = model(imgs)              # [B,K,64,64] by default
            logits, targs = _resize_to(logits, targs)
            loss = criterion(logits, targs)
            loss.backward()
            optim_.step()
            run += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        tr_loss = run / max(seen, 1)

        # val
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for imgs, targs in val_loader:
                imgs  = imgs.to(device).float()
                targs = targs.to(device).float()
                logits = model(imgs)
                logits, targs = _resize_to(logits, targs)
                loss = criterion(logits, targs)
                tot += loss.item() * imgs.size(0)
                n   += imgs.size(0)
        val_loss = tot / max(n, 1)
        print(f"[HM Abl][{ep:02d}/{epochs}] train={tr_loss:.4f} val={val_loss:.4f}")

        best_val = min(best_val, val_loss)
    return best_val


@torch.no_grad()
def eval_pixel_metrics_heatmap(model, loader, device, up_factor):
    model.eval()
    errs, pcks_5 = [], []
    for imgs, tgt in loader:
        B = imgs.size(0)
        imgs = imgs.to(device).float()
        # decode preds
        logits = model(imgs)                       
        pred_xy = extract_keypoints_from_heatmaps(logits, up_factor=up_factor)  
        tgt = tgt.to(device).float() 
        tgt_xy = extract_keypoints_from_heatmaps(tgt, up_factor=IMAGE_SIZE // tgt.shape[-1])
        errs.append(mean_pixel_error(pred_xy, tgt_xy))
        pcks_5.append(pck(pred_xy, tgt_xy, size=IMAGE_SIZE, alpha=0.05))
    return float(np.mean(errs)), float(np.mean(pcks_5))

@torch.no_grad()
def eval_pixel_metrics_regression(model, loader, device):
    model.eval()
    errs, pcks_5 = [], []
    for imgs, tgt_reg in loader:
        imgs = imgs.to(device).float()
        gt_xy = (tgt_reg.view(-1, NUM_KPS, 2) * float(IMAGE_SIZE)).to(imgs.device)
        pred  = model(imgs)  # [B,2K] in [0,1]
        pred_xy = (pred.view(-1, NUM_KPS, 2) * float(IMAGE_SIZE))
        errs.append(mean_pixel_error(pred_xy, gt_xy))
        pcks_5.append(pck(pred_xy, gt_xy, size=IMAGE_SIZE, alpha=0.05))
    return float(np.mean(errs)), float(np.mean(pcks_5))


def ablation_study():
    """
    Conduct ablation studies on key hyperparameters.
    
    Experiments to run:
    1. Effect of heatmap resolution (32x32 vs 64x64 vs 128x128)
    2. Effect of Gaussian sigma (1.0, 2.0, 3.0, 4.0)
    3. Effect of skip connections (with vs without)
    """
    # Run experiments and save results
    dev = get_device()
    print("Device:", dev)
    results = {"heatmap_res": [], "sigma": [], "skip": []}

    def build_loaders(heatmap_size, sigma, batch=32):
        tr = KeypointDataset(str(TRAIN_IMG_DIR), str(TRAIN_ANN),
                             output_type="heatmap", heatmap_size=heatmap_size, sigma=sigma)
        va = KeypointDataset(str(VAL_IMG_DIR),   str(VAL_ANN),
                             output_type="heatmap", heatmap_size=heatmap_size, sigma=sigma)
        return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
                DataLoader(va, batch_size=batch, shuffle=False, num_workers=0))

    # (1) Heatmap resolution
    for hm_size in [32, 64, 128]:
        train_loader, val_loader = build_loaders(hm_size, sigma=2.0)
        model = HeatmapNet(num_keypoints=NUM_KPS).to(dev)
        best_val = train_heatmap_once(model, train_loader, val_loader, dev,
                                      epochs=6, lr=1e-3, target_hm_size=hm_size)
        mpe, pck5 = eval_pixel_metrics_heatmap(model, val_loader, dev, up_factor=IMAGE_SIZE // hm_size)
        results["heatmap_res"].append({"heatmap_size": hm_size, "best_val": best_val,
                                       "mpe_px": mpe, "pck@0.05": pck5})

    # (2) Gaussian sigma
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        train_loader, val_loader = build_loaders(heatmap_size=64, sigma=sigma)
        model = HeatmapNet(num_keypoints=NUM_KPS).to(dev)
        best_val = train_heatmap_once(model, train_loader, val_loader, dev,
                                      epochs=6, lr=1e-3, target_hm_size=64)
        mpe, pck5 = eval_pixel_metrics_heatmap(model, val_loader, dev, up_factor=IMAGE_SIZE // 64)
        results["sigma"].append({"sigma": sigma, "best_val": best_val,
                                 "mpe_px": mpe, "pck@0.05": pck5})

    # (3) Skip connections (heatmap net)
    # with skips
    tr_w, va_w = build_loaders(heatmap_size=64, sigma=2.0)
    model_w = HeatmapNet(num_keypoints=NUM_KPS).to(dev)
    best_val_w = train_heatmap_once(model_w, tr_w, va_w, dev, epochs=6, lr=1e-3, target_hm_size=64)
    mpe_w, pck5_w = eval_pixel_metrics_heatmap(model_w, va_w, dev, up_factor=IMAGE_SIZE // 64)

    # without skips
    tr_n, va_n = build_loaders(heatmap_size=64, sigma=2.0)
    model_n = HeatmapNetNoSkip(num_keypoints=NUM_KPS).to(dev)
    best_val_n = train_heatmap_once(model_n, tr_n, va_n, dev, epochs=6, lr=1e-3, target_hm_size=64)
    mpe_n, pck5_n = eval_pixel_metrics_heatmap(model_n, va_n, dev, up_factor=IMAGE_SIZE // 64)

    results["skip"].append({
        "with_skip":   {"best_val": best_val_w, "mpe_px": mpe_w, "pck@0.05": pck5_w},
        "without_skip":{"best_val": best_val_n, "mpe_px": mpe_n, "pck@0.05": pck5_n},
    })

    ablation_path = RESULTS_DIR / "ablations.json"
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", ablation_path)

    xs = [r["heatmap_size"] for r in results["heatmap_res"]]
    ys = [r["best_val"] for r in results["heatmap_res"]]
    fig_path = VIS_DIR / "ablation_heatmap_size_vs_val.png"
    ensure_parent(fig_path)
    plt.figure(figsize=(5,3))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Heatmap size")
    plt.ylabel("Best val loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150); plt.close()
    print("Saved:", fig_path)

@torch.no_grad()
def analyze_failure_cases(top_k=6):
    """
    Identify and visualize failure cases.
    
    Find examples where:
    1. Heatmap succeeds but regression fails
    2. Regression succeeds but heatmap fails
    3. Both methods fail
    """
    dev = get_device()
    print("Device:", dev)

    test_ds = KeypointDataset(str(TEST_IMG_DIR), str(TEST_ANN),
                              output_type="regression", heatmap_size=DEFAULT_HM, sigma=2.0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    hm_model = HeatmapNet(NUM_KPS).to(dev)
    rg_model = RegressionNet(NUM_KPS).to(dev)

    hm_w = RESULTS_DIR / "heatmap_model.pth"
    rg_w = RESULTS_DIR / "regression_model.pth"
    if hm_w.exists(): hm_model.load_state_dict(torch.load(hm_w, map_location=dev))
    if rg_w.exists(): rg_model.load_state_dict(torch.load(rg_w, map_location=dev))

    hm_model.eval(); rg_model.eval()

    imgs_all, gt_all, hm_xy_all, rg_xy_all = [], [], [], []

    for imgs, tgt_reg in test_loader:
        B = imgs.size(0)
        imgs = imgs.to(dev).float()
        gt_xy = (tgt_reg.view(B, NUM_KPS, 2) * float(IMAGE_SIZE)).to(dev)

        # heatmap decode
        hm_logits = hm_model(imgs)
        hm_xy = extract_keypoints_from_heatmaps(hm_logits, up_factor=IMAGE_SIZE // hm_logits.shape[-1])

        # regression decode
        rg_out = rg_model(imgs)
        rg_xy = (rg_out.view(B, NUM_KPS, 2) * float(IMAGE_SIZE))

        imgs_all.append(imgs.cpu())
        gt_all.append(gt_xy.cpu())
        hm_xy_all.append(hm_xy.cpu())
        rg_xy_all.append(rg_xy.cpu())

    imgs = torch.cat(imgs_all, dim=0)   
    gt   = torch.cat(gt_all,   dim=0)   
    hm_xy= torch.cat(hm_xy_all,dim=0)   
    rg_xy= torch.cat(rg_xy_all,dim=0)  

    # per-sample mean errors
    err_h = torch.linalg.norm(hm_xy - gt, dim=-1).mean(dim=-1)  
    err_r = torch.linalg.norm(rg_xy - gt, dim=-1).mean(dim=-1) 

    thr = 0.05 * IMAGE_SIZE

    mask_h_win = (err_h < thr) & (err_r >= thr)
    mask_r_win = (err_r < thr) & (err_h >= thr)
    mask_both_fail = (err_h >= thr) & (err_r >= thr)

    def save_examples(mask, prefix):
        idxs = torch.nonzero(mask).flatten().tolist()[:top_k]
        for i, idx in enumerate(idxs):
            visualize_predictions(imgs[idx], hm_xy[idx], gt[idx], VIS_DIR / f"{prefix}_HM_{i}.png")
            visualize_predictions(imgs[idx], rg_xy[idx], gt[idx], VIS_DIR / f"{prefix}_RG_{i}.png")

    save_examples(mask_h_win, "case_heatmap_wins")
    save_examples(mask_r_win, "case_regression_wins")
    save_examples(mask_both_fail, "case_both_fail")

    print("Saved failure cases to:", VIS_DIR)


def main():
    # Run ablations 
    ablation_study()
    # Analyze failure cases with trained models from results/
    analyze_failure_cases(top_k=6)


if __name__ == "__main__":
    main()

    