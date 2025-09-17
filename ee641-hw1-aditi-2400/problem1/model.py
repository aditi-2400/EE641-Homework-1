import torch
import torch.nn as nn

class ConvBNRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    

class DetectionHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
       super().__init__()
       self.conv3x3 = nn.Conv2d(in_c,in_c,kernel_size=3,stride=1,padding=1,bias=True)
       out_c = num_anchors * (5 + num_classes) 
       self.conv1x1 = nn.Conv2d(in_c,out_c,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Initialize the multi-scale detector.
        
        Args:
            num_classes: Number of object classes (not including background)
            num_anchors: Number of anchors per spatial location
        """

        
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.stem1 = ConvBNRelu(in_c=3, out_c=32, stride=1, padding=1)
        self.stem2 = ConvBNRelu(in_c=32, out_c=64, stride=2, padding=1)

        self.block2 = ConvBNRelu(in_c=64, out_c=128, stride=2, padding=1) # Scale 1
        self.block3 = ConvBNRelu(in_c=128, out_c=256, stride=2, padding=1) # Scale 2
        self.block4 = ConvBNRelu(in_c=256, out_c=512, stride=2, padding=1) # Scale 3

        self.head_s1 = DetectionHead(in_c=128, num_anchors=num_anchors, num_classes=num_classes)  
        self.head_s2 = DetectionHead(in_c=256, num_anchors=num_anchors, num_classes=num_classes)  
        self.head_s3 = DetectionHead(in_c=512, num_anchors=num_anchors, num_classes=num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 3, 224, 224]
            
        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """
        x = self.stem1(x)
        x = self.stem2(x)
        scale1_feat = self.block2(x)
        scale2_feat = self.block3(scale1_feat)
        scale3_feat = self.block4(scale2_feat)

        scale1_out = self.head_s1(scale1_feat)
        scale2_out = self.head_s2(scale2_feat)
        scale3_out = self.head_s3(scale3_feat)

        return [scale1_out, scale2_out, scale3_out]