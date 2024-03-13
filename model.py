import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial convolution block
        self.inc = DoubleConv(n_channels, 64)
        
        # Downsample
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        
        # Add Transposed Convolution for Upsampling
        self.up_tconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_tconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_tconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Upsample
        self.up1 = DoubleConv(512 + 256, 256)  # Adjusted in_channels to 512 + 256 for the concatenated features
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Downsampling
        x1 = self.inc(x)
        x2 = F.max_pool2d(self.down1(x1), 2)
        x3 = F.max_pool2d(self.down2(x2), 2)
        x4 = F.max_pool2d(self.down3(x3), 2)
        
        # Upsampling with Transposed Convolution
        x = self.up_tconv1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        
        x = self.up_tconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        
        x = self.up_tconv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)
        
        logits = self.outc(x)
        return logits
