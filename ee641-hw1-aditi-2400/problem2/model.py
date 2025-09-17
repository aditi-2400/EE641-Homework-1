import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeconvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=2, s=2, p=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(ConvBNReLU(1, 32), nn.MaxPool2d(2))     
        self.c2 = nn.Sequential(ConvBNReLU(32, 64), nn.MaxPool2d(2))    
        self.c3 = nn.Sequential(ConvBNReLU(64, 128), nn.MaxPool2d(2))   
        self.c4 = nn.Sequential(ConvBNReLU(128, 256), nn.MaxPool2d(2))  

    def forward(self, x):
        f1 = self.c1(x)   
        f2 = self.c2(f1)  
        f3 = self.c3(f2)  
        f4 = self.c4(f3)  
        return f1, f2, f3, f4

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the heatmap regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Skip connections between encoder and decoder
        super().__init__()
        self.enc = Encoder()

        self.deconv4 = DeconvBNReLU(256, 128)   
        self.dec3 = nn.Sequential(
            ConvBNReLU(256, 128),
            ConvBNReLU(128, 128),
            DeconvBNReLU(128, 64)                
        )
        self.dec2 = nn.Sequential(
            ConvBNReLU(128, 64),
            DeconvBNReLU(64, 32)                 
        )
        self.head = nn.Conv2d(32, num_keypoints, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            heatmaps: Tensor of shape [batch, num_keypoints, 64, 64]
        """
        f1, f2, f3, f4 = self.enc(x)

        up4 = self.deconv4(f4)               
        x3u = torch.cat([up4, f3], dim=1)    

        up3 = self.dec3(x3u)                 
        x2u = torch.cat([up3, f2], dim=1)    

        up2 = self.dec2(x2u)                 

        heatmaps = self.head(up2)           
        return heatmaps


class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.enc = Encoder()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, 64),  nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(64, num_keypoints * 2)
        )

    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        _, _, _, f4 = self.enc(x)            
        z = self.gap(f4).flatten(1)           
        coords = torch.sigmoid(self.mlp(z))   
        return coords
