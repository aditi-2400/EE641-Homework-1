import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "datasets" / "keypoints"   

TRAIN_IMG_DIR = DATA_DIR / "train"
TRAIN_ANN     = DATA_DIR / "train_annotations.json"
VAL_IMG_DIR   = DATA_DIR / "val"
VAL_ANN       = DATA_DIR / "val_annotations.json"

RESULTS_DIR   = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_WEIGHTS   = RESULTS_DIR / "heatmap_model.pth"
REGRESSION_WEIGHTS= RESULTS_DIR / "regression_model.pth"
LOG_PATH          = RESULTS_DIR / "training_log.json"

EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-3
HEATMAP_SIZE = 64

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def evaluate_heatmap(model, loader, device):
    """MSE between sigmoid(pred_heatmap) and target heatmap."""
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    count = 0
    for images, targets in loader:
        images    = images.to(device)
        targets = targets.to(device)  
        logits  = model(images)         
        preds   = torch.sigmoid(logits)
        loss    = criterion(preds, targets)
        total  += loss.item() * images.size(0)
        count  += images.size(0)
    return total / max(count, 1)

@torch.no_grad()
def evaluate_regression(model, loader, device):
    """MSE between predicted normalized coords [B, 2K] and target [B, 2K]."""
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    count = 0
    for images, targets in loader:
        images    = images.to(device)
        targets = targets.to(device)  
        preds   = model(images)         
        loss    = criterion(preds, targets)
        total  += loss.item() * images.size(0)
        count  += images.size(0)
    return total / max(count, 1)

def train_heatmap_model(model, train_loader, val_loader, device, save_path, num_epochs=30):
    """
    Train the heatmap-based model.
    
    Uses MSE loss between predicted and target heatmaps.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in train_loader:
            images    = images.to(device)
            targets = targets.to(device)  

            optimizer.zero_grad()
            logits = model(images)          
            preds  = torch.sigmoid(logits)
            loss   = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)
            seen    += images.size(0)

        train_loss = running / max(seen, 1)
        val_loss   = evaluate_heatmap(model, val_loader, device)

        history.append({"epoch": epoch, "heatmap_train_loss": float(train_loss), "heatmap_val_loss": float(val_loss)})
        print(f"[Heatmap][{epoch:02d}/{num_epochs}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val and save_path is not None:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

    return history


def train_regression_model(model, train_loader, val_loader, device, save_path, num_epochs=30):
    """
    Train the direct regression model.
    
    Uses MSE loss between predicted and target coordinates.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in train_loader:
            images    = images.to(device)
            targets = targets.to(device)  

            optimizer.zero_grad()
            preds = model(images)           
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)
            seen    += images.size(0)

        train_loss = running / max(seen, 1)
        val_loss   = evaluate_regression(model, val_loader, device)

        history.append({"epoch": epoch, "reg_train_loss": float(train_loss), "reg_val_loss": float(val_loss)})
        print(f"[Regress][{epoch:02d}/{num_epochs}]  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val and save_path is not None:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

    return history

def main():
    device = get_device()
    print("Device:", device)

    train_dataset_hm = KeypointDataset(str(TRAIN_IMG_DIR), str(TRAIN_ANN),
                                  output_type="heatmap", heatmap_size=HEATMAP_SIZE, sigma=2.0)
    val_dataset_hm   = KeypointDataset(str(VAL_IMG_DIR),   str(VAL_ANN),
                                  output_type="heatmap", heatmap_size=HEATMAP_SIZE, sigma=2.0)

    train_loader_hm = DataLoader(train_dataset_hm, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader_hm   = DataLoader(val_dataset_hm,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_dataset_rg = KeypointDataset(str(TRAIN_IMG_DIR), str(TRAIN_ANN),
                                  output_type="regression", heatmap_size=HEATMAP_SIZE, sigma=2.0)
    val_dataset_rg   = KeypointDataset(str(VAL_IMG_DIR),   str(VAL_ANN),
                                  output_type="regression", heatmap_size=HEATMAP_SIZE, sigma=2.0)

    train_loader_rg = DataLoader(train_dataset_rg, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader_rg   = DataLoader(val_dataset_rg,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    heatmap_model   = HeatmapNet(num_keypoints=5).to(device)
    regression_model= RegressionNet(num_keypoints=5).to(device)

    # --- Train ---
    log = {"heatmap": [], "regression": []}

    log["heatmap"] = train_heatmap_model(
        heatmap_model, train_loader_hm, val_loader_hm,
        device=device, num_epochs=EPOCHS, save_path=HEATMAP_WEIGHTS
    )

    log["regression"] = train_regression_model(
        regression_model, train_loader_rg, val_loader_rg,
        device=device, num_epochs=EPOCHS, save_path=REGRESSION_WEIGHTS
    )

    # --- Save combined log ---
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print("Saved weights:", HEATMAP_WEIGHTS, REGRESSION_WEIGHTS)
    print("Saved log:", LOG_PATH)

if __name__ == '__main__':
    main()