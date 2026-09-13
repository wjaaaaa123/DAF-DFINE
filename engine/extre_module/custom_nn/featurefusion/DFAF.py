
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else 0
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFAF(nn.Module):
    """
    DFAF: Detail-Frequency Alignment Fusion

    A fully-convolutional second innovation inspired by the *problem*
    addressed by Fourier Angle Alignment: directional inconsistency
    between high- and low-level neck features.

    Unlike FAAFusion, this module does NOT explicitly estimate Fourier
    angles or rotate patches. Instead it:
      1) decomposes each feature into low-frequency base + high-frequency detail;
      2) measures detail-frequency discrepancy;
      3) routes the high-level detail through H/V/context directional paths;
      4) aligns high-level detail to the low-level feature;
      5) performs adaptive two-branch fusion.

    This avoids patch-wise FFT/rotation and is easier to deploy / profile.

    Input:
        x = [x_high, x_low]
        both usually have C=256.
        x_high may be lower resolution; it is resized automatically.

    Output:
        [B, C, H_low, W_low]

    Parser-friendly signature:
        DFAF(channels=256, reduction=4)
    """

    def __init__(self, channels=256, reduction=4):
        super().__init__()
        c = channels
        mid = max(c // reduction, 16)

        self.high_proj = ConvBNAct(c, c, 1, 1, act=False)
        self.low_proj = ConvBNAct(c, c, 1, 1, act=False)

        # Directional alignment paths operating on high-frequency detail.
        self.dir_h = nn.Conv2d(
            c, c, kernel_size=(1, 5), stride=1,
            padding=(0, 2), groups=c, bias=False
        )
        self.dir_v = nn.Conv2d(
            c, c, kernel_size=(5, 1), stride=1,
            padding=(2, 0), groups=c, bias=False
        )
        self.dir_c = nn.Conv2d(
            c, c, kernel_size=3, stride=1,
            padding=1, groups=c, bias=False
        )

        # Routing weights are generated from channel-compressed
        # low/high-frequency discrepancies.
        self.route = nn.Sequential(
            nn.Conv2d(4, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 3, 1, 1, 0, bias=True)
        )

        # Spatial fusion gate.
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(3, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            ConvBNAct(c, c, 3, 1, g=c, act=True),
            ConvBNAct(c, c, 1, 1, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.act = nn.SiLU(inplace=True)

    @staticmethod
    def _split_frequency(x):
        # 3x3 local average approximates the low-frequency component.
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        return low, high

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DFAF expects [x_high, x_low].")

        x_high, x_low = x
        if x_high.shape[-2:] != x_low.shape[-2:]:
            x_high = F.interpolate(
                x_high, size=x_low.shape[-2:],
                mode="bilinear", align_corners=False
            )

        h = self.high_proj(x_high)
        l = self.low_proj(x_low)

        h_low, h_high = self._split_frequency(h)
        l_low, l_high = self._split_frequency(l)

        # Frequency/detail discrepancy descriptors.
        d_low = (h_low - l_low).abs().mean(1, keepdim=True)
        d_high = (h_high - l_high).abs().mean(1, keepdim=True)
        h_energy = h_high.abs().mean(1, keepdim=True)
        l_energy = l_high.abs().mean(1, keepdim=True)

        route_input = torch.cat(
            [d_low, d_high, h_energy, l_energy], dim=1
        )
        route = torch.softmax(self.route(route_input), dim=1)

        # Directionally re-express high-level high-frequency detail.
        aligned_detail = (
            route[:, 0:1] * self.dir_h(h_high) +
            route[:, 1:2] * self.dir_v(h_high) +
            route[:, 2:3] * self.dir_c(h_high)
        )

        # Align high-level representation toward the low-level
        # frequency/detail structure.
        h_aligned = h_low + h_high + self.gamma * aligned_detail

        # Dynamic spatial fusion, conditioned on discrepancy.
        fusion_desc = torch.cat(
            [
                (h_aligned - l).abs().mean(1, keepdim=True),
                h_aligned.abs().mean(1, keepdim=True),
                l.abs().mean(1, keepdim=True),
            ],
            dim=1,
        )
        alpha = self.fuse_gate(fusion_desc)

        fused = alpha * l + (1.0 - alpha) * h_aligned
        fused = fused + self.out(fused)
        return self.act(fused)
