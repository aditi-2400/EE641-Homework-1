import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0, image_size=128):
        """
        Initialize the keypoint dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        assert output_type in ("heatmap", "regression")
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.image_size = image_size

        with open(annotation_file, "r") as f:
            data = json.load(f)
        # Load annotations
        self.images = data["images"]
        self.num_keypoints = data["num_keypoints"]

    def __len__(self):
        return len(self.images)
    
    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap
            
        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        # For each keypoint:
        # 1. Create 2D gaussian centered at keypoint location
        # 2. Handle boundary cases
        heatmaps = np.zeros((len(keypoints), height, width), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        for k, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
            heatmaps[k] = gaussian
        return torch.tensor(heatmaps)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        entry = self.images[idx]

        # Load image (resize to 128x128 grayscale)
        img_path = os.path.join(self.image_dir, entry["file_name"])
        image = Image.open(img_path).convert("L")
        image = image.resize((self.image_size, self.image_size))
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0

        # Keypoints
        keypoints = np.array(entry["keypoints"], dtype=np.float32)

        if self.output_type == "heatmap":
            # Scale keypoints to heatmap size
            scale_x = self.heatmap_size / self.image_size
            scale_y = self.heatmap_size / self.image_size
            kp_scaled = [(x * scale_x, y * scale_y) for (x, y) in keypoints]
            targets = self.generate_heatmap(kp_scaled, self.heatmap_size, self.heatmap_size)
        else:  # regression
            # Normalize to [0,1]
            kp_norm = keypoints / self.image_size
            targets = torch.tensor(kp_norm.flatten(), dtype=torch.float32)

        return image, targets