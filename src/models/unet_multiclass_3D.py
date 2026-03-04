# unet_multiclass_3D.py
import torch
import torch.nn as nn

from unet_parts_3D import DoubleConv3D, DownSample3D, UpSample3D


class UNet3D(nn.Module):
    """
    UNet 3D para segmentación multiclase (Couinaud 0..8).

    Input : [B, 1, D, H, W]  (en tu caso D=32)
    Output: [B, 9, D, H, W]
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 9, base_filters: int = 32, norm: str = "in"):
        super().__init__()

        self.down_convolution_1 = DownSample3D(in_channels, base_filters, norm=norm)
        self.down_convolution_2 = DownSample3D(base_filters, base_filters * 2, norm=norm)
        self.down_convolution_3 = DownSample3D(base_filters * 2, base_filters * 4, norm=norm)
        self.down_convolution_4 = DownSample3D(base_filters * 4, base_filters * 8, norm=norm)

        # Bottleneck (dropout = 0.0)
        self.bottle_neck = DoubleConv3D(base_filters * 8, base_filters * 16, norm=norm, dropout=0.0)

        self.up_convolution_1 = UpSample3D(base_filters * 16, base_filters * 8, norm=norm)
        self.up_convolution_2 = UpSample3D(base_filters * 8, base_filters * 4, norm=norm)
        self.up_convolution_3 = UpSample3D(base_filters * 4, base_filters * 2, norm=norm)
        self.up_convolution_4 = UpSample3D(base_filters * 2, base_filters, norm=norm)

        self.out = nn.Conv3d(in_channels=base_filters, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 1, 32, 512, 512)
    model = UNet3D(in_channels=1, num_classes=9, base_filters=16, norm="in")
    y = model(x)
    print("Output:", y.shape)
