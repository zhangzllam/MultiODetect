import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import math

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class LSKattention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class ResidualBlock(nn.Module):   #RHLblock
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #Down_wt(out_ch, out_ch),
            LSKattention(out_ch),
            #LSKblock(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Down_wt(out_ch, out_ch),
        )

        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1) if in_ch != out_ch else nn.Identity()
        self.wtpool = Down_wt(out_ch, out_ch)

    def forward(self, x):
        return self.block(x) + self.wtpool(self.shortcut(x))  # Residual Connection

class HWD_LSKNetv1(nn.Module):
    def __init__(self):
        super(HWD_LSKNetv1, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regression = nn.Sequential(
            nn.Linear(512, 1280),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1280, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.regression(x)
        return x.squeeze(-1)

class HWD_LSKNetv2(nn.Module):
    def __init__(self, num_classes):
        super(HWD_LSKNetv2, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 1280),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1280, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x