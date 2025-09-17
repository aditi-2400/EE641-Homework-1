import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
import numpy as np

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        # Load and parse annotations
        # Store image paths and corresponding annotations
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        with open(annotation_file) as f:
            coco = json.load(f)
            self.images = coco["images"]
            annotations = coco["annotations"]
            categories = coco["categories"]

            self.cat_id_to_label = {cat["id"]:cat["id"] for cat in categories}
            self.label_to_name = {cat["id"]:cat["name"] for cat in categories}  

            self.img_ann = {img["id"]:[] for img in self.images}
            for ann in annotations:
                self.img_ann[ann["image_id"]].append(ann)

    
    def __len__(self):
        return len(self.images)
        """Return the total number of samples."""
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        info = self.images[idx]
        img_path = self.image_dir/info["file_name"]
        img = Image.open(img_path).convert("RGB")

        ann = self.img_ann.get(info["id"],[])
        boxes, labels = [], []
        for a in ann:
            x1, y1, x2, y2 = a["bbox"]
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[a["category_id"]])
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "id": info["id"]
        }
        if self.transform is None:
            arr = np.asarray(img, dtype=np.float32) / 255.0  
            arr = np.transpose(arr, (2, 0, 1))               
            img = torch.from_numpy(arr)
        else:
            img, target = self.transform(img, target)
        
        return img, target


def detection_collate(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)