import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time
from dataset import ShapeDetectionDataset, detection_collate
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors


TRAIN_IMG_DIR = "../datasets/detection/train"
TRAIN_ANN     = "../datasets/detection/train_annotations.json"
VAL_IMG_DIR   = "../datasets/detection/val"
VAL_ANN       = "../datasets/detection/val_annotations.json"


RESULTS_DIR = "results"
BEST_MODEL_PATH = f"{RESULTS_DIR}/best_model.pth"
LOG_PATH        = f"{RESULTS_DIR}/training_log.json"

IMAGE_SIZE = 224
FEATURE_MAP_SIZES = [(56, 56), (28, 28), (14, 14)]
ANCHOR_SCALES = [
    [16, 24, 32],     # 56x56
    [48, 64, 96],     # 28x28
    [96, 128, 192],   # 14x14
]

NUM_CLASSES = 3
NUM_ANCHORS = 3   
BATCH_SIZE = 16
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 50
RESULTS_DIR = "results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.json")
SEED = 42

def to_device_targets(targets, device):
    """Move target dicts to device."""
    out = []
    for t in targets:
        out.append({
            "boxes":  t["boxes"].to(device),
            "labels": t["labels"].to(device),
            "image_id": t.get("image_id", torch.tensor([-1])).to(device)
        })
    return out

def train_epoch(model, dataloader, criterion, optimizer, device, anchors_per_level):
    """Train for one epoch."""
    model.train()
    running_loss = {"obj":0, "cls":0, "loc":0, "tot":0}
    count = 0
    for images,targets in dataloader:
        images = [image.to(device) for image in images]
        targets = to_device_targets(targets, device)
        optimizer.zero_grad()

        x = torch.stack(images, dim=0)
        preds = model(x)
        # with torch.no_grad():
        #     per = 5 + NUM_CLASSES
        #     for i, p in enumerate(preds):
        #         B, Ctot, H, W = p.shape
        #         assert Ctot % per == 0, "Head channels must be A*(5+C)"
        #         A_from_head = Ctot // per
        #         N_pred = H * W * A_from_head
        #         N_anchors = anchors_per_level[i].shape[0]
        #         print(f"[scale {i}] Ctot={Ctot}, HxW={H}x{W}, A_from_head={A_from_head}, "
        #             f"N_pred={N_pred}, N_anchors={N_anchors}")
        loss_dict = criterion(preds, targets, anchors_per_level)
        loss = loss_dict["loss_tot"]
        loss.backward()
        optimizer.step()
        for k in running_loss:
            running_loss[k] += loss_dict[f"loss_{k}"].item()
        count += 1

    for k in running_loss:
        running_loss[k] /= max(count, 1)
    return running_loss
    
@torch.no_grad()
def validate(model, dataloader, criterion, device, anchors_per_level):
    """Validate the model."""
    model.eval()
    running_loss = {"obj":0, "cls":0, "loc":0, "tot":0}
    count = 0
    for images,targets in dataloader:
        images = [image.to(device) for image in images]
        targets = to_device_targets(targets, device)
        x = torch.stack(images, dim=0)
        preds = model(x)

        loss_dict = criterion(preds, targets, anchors_per_level)
        for k in running_loss:
            running_loss[k] += loss_dict[f"loss_{k}"].item()
        count += 1

    for k in running_loss:
        running_loss[k] /= max(count, 1)
    return running_loss
    # Validation loop
    

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    torch.manual_seed(SEED)

    train_dataset = ShapeDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANN)
    val_dataset = ShapeDetectionDataset(VAL_IMG_DIR, VAL_ANN)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=detection_collate)
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=detection_collate)
    
    model = MultiScaleDetector(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        pred_shapes = [(p.shape[2], p.shape[3]) for p in model(dummy)]

    anchors_per_level = generate_anchors(pred_shapes, ANCHOR_SCALES, image_size=IMAGE_SIZE)
    anchors_per_level = [a.to(device) for a in anchors_per_level]

    criterion = DetectionLoss(num_classes=NUM_CLASSES).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val = float("inf")
    log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device, anchors_per_level)
        val_metrics   = validate(model, val_dataloader, criterion, device, anchors_per_level)
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        
        val_total = val_metrics["tot"]
        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        record = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "train": {k: round(v, 6) for k, v in train_metrics.items()},
            "val":   {k: round(v, 6) for k, v in val_metrics.items()},
            "best_val_loss": round(best_val, 6)
        }
        log.append(record)

        print(f"[{epoch:03d}/{NUM_EPOCHS}] "
              f"train_total={record['train']['tot']:.4f} "
              f"val_total={record['val']['tot']:.4f} "
              f"(obj {record['val']['obj']:.4f} | cls {record['val']['cls']:.4f} | loc {record['val']['loc']:.4f}) "
              f"lr={record['lr']:.5f} time={record['time_sec']}s")

        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    print("Training complete.")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Training log saved to: {LOG_PATH}")
    # Initialize dataset, model, loss, optimizer
    # Training loop with logging
    # Save best model and training log
    

if __name__ == '__main__':
    main()
