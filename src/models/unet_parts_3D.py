# unet_parts_3D.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    (Conv3d -> Norm -> ReLU) x2
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "in", dropout: float = 0.0):
        super().__init__()

        if norm == "bn":
            Norm = nn.BatchNorm3d
        elif norm == "in":
            Norm = nn.InstanceNorm3d
        elif norm == "gn":
            groups = 8 if out_ch >= 8 else 1
            Norm = lambda c: nn.GroupNorm(groups, c)
        else:
            raise ValueError("norm debe ser 'bn', 'in' o 'gn'")

        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        ]

        # Dropout removido (queda disponible si algún día querés reactivarlo)
        if dropout and dropout > 0:
            layers.append(nn.Dropout3d(p=float(dropout)))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownSample3D(nn.Module):
    """
    Devuelve:
      - down: feature map para skip connection
      - p   : pooled para seguir bajando
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "in"):
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch, norm=norm, dropout=0.0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample3D(nn.Module):
    """
    upsample + concat con skip + DoubleConv
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "in"):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch, norm=norm, dropout=0.0)

    def forward(self, x, skip):
        x = self.up(x)

        diffD = skip.size(2) - x.size(2)
        diffH = skip.size(3) - x.size(3)
        diffW = skip.size(4) - x.size(4)

        x = F.pad(
            x,
            [
                diffW // 2, diffW - diffW // 2,
                diffH // 2, diffH - diffH // 2,
                diffD // 2, diffD - diffD // 2,
            ]
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x
