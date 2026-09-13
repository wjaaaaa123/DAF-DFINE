
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = ((k - 1) * d) // 2 if isinstance(k, int) else 0
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DHEC(nn.Module):
    """
    DHEC: Direction-aware High-order Edge Convolution

    Working design for D-FINE aluminium defect detection.

    Motivation:
    1) polynomial/high-order spatial interaction:
       fine and context branches are multiplied element-wise;
    2) direction-aware edge guidance:
       horizontal, vertical and diagonal responses generate a spatial gate;
    3) stable first-order residual path:
       keeps ordinary local/context information to avoid relying only on
       multiplicative high-order responses.

    Signature is compatible with the project's c1/c2 parser:
        DHEC(c1, c2, expansion=1.0, context_kernel=5, dilation=2)
    """

    def __init__(self, c1, c2, expansion=1.0, context_kernel=5, dilation=2):
        super().__init__()
        hidden = _make_divisible(c2 * expansion, 8)

        self.in_proj = ConvBNAct(c1, hidden, 1, 1)

        # Fine / context pair for second-order interaction.
        self.fine = ConvBNAct(hidden, hidden, 3, 1, g=hidden, act=False)
        self.context = ConvBNAct(
            hidden, hidden, context_kernel, 1,
            g=hidden, d=dilation, act=False
        )

        # Directional edge branches. These are learnable and lightweight.
        self.edge_h = nn.Conv2d(
            hidden, hidden, kernel_size=(1, 5),
            stride=1, padding=(0, 2), groups=hidden, bias=False
        )
        self.edge_v = nn.Conv2d(
            hidden, hidden, kernel_size=(5, 1),
            stride=1, padding=(2, 0), groups=hidden, bias=False
        )
        self.edge_d1 = nn.Conv2d(
            hidden, hidden, kernel_size=3,
            stride=1, padding=1, groups=hidden, bias=False
        )
        self.edge_d2 = nn.Conv2d(
            hidden, hidden, kernel_size=3,
            stride=1, padding=2, dilation=2, groups=hidden, bias=False
        )

        # Spatial edge gate: collapse directional evidence to one map.
        self.edge_gate = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

        # Learnable scale starts small so the module initially behaves
        # mostly like the stable first-order path.
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.refine = nn.Sequential(
            ConvBNAct(hidden, hidden, 3, 1, g=hidden, act=True),
            ConvBNAct(hidden, c2, 1, 1, act=False),
        )

        self.shortcut = (
            nn.Identity() if c1 == c2
            else ConvBNAct(c1, c2, 1, 1, act=False)
        )
        self.out_act = nn.SiLU(inplace=True)

    def _edge_descriptor(self, x):
        eh = self.edge_h(x).abs().mean(1, keepdim=True)
        ev = self.edge_v(x).abs().mean(1, keepdim=True)
        ed1 = self.edge_d1(x).abs().mean(1, keepdim=True)
        ed2 = self.edge_d2(x).abs().mean(1, keepdim=True)
        return torch.cat([eh, ev, ed1, ed2], dim=1)

    def forward(self, x):
        identity = self.shortcut(x)

        z = self.in_proj(x)
        f = self.fine(z)
        c = self.context(z)

        # First-order stable component.
        first_order = 0.5 * (f + c)

        # High-order polynomial interaction.
        high_order = f * c

        # Direction-aware edge gate.
        gate = self.edge_gate(self._edge_descriptor(z))

        y = z + first_order + self.beta * gate * high_order
        y = self.refine(y)
        return self.out_act(identity + y)
