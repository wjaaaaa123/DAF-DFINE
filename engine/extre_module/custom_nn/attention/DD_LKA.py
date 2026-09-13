import torch
import torch.nn as nn
import torchvision
from engine.core.workspace import register


@register()
class DD_LKA(nn.Module):

    def __init__(
            self,
            in_channels,
            groups,
            kernel_size=(3,3),
            padding=1,
            stride=1,
            dilation=1
    ):
        super().__init__()

        self.offset_net = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation
        )

        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=False
        )


    def forward(self,x):

        offset = self.offset_net(x)

        out = self.deform_conv(
            x,
            offset
        )

        return out
class DeformConv(nn.Module):

    def __init__(
        self,
        in_channels,
        groups,
        kernel_size=(3,3),
        padding=1,
        stride=1,
        dilation=1,
        bias=True
    ):

        super().__init__()


        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=True
        )


        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False
        )


    def forward(self,x):

        offsets = self.offset_net(x)

        out = self.deform_conv(
            x,
            offsets
        )

        return out


class DD_LKA(nn.Module):
    """
    Defect-aware Deformable Large Kernel Attention

    Based on:
    Deformable LKA WACV2024
    """

    def __init__(self, dim):

        super().__init__()


        # -------------------------
        # Original Deformable LKA
        # -------------------------

        self.conv0 = DeformConv(
            dim,
            kernel_size=(5,5),
            padding=2,
            groups=dim
        )


        self.conv_spatial = DeformConv(
            dim,
            kernel_size=(7,7),
            padding=9,
            dilation=3,
            groups=dim
        )


        self.conv1 = nn.Conv2d(
            dim,
            dim,
            1
        )


        # -------------------------
        # Defect Prior Branch
        # -------------------------

        self.defect = nn.Sequential(

            nn.Conv2d(
                dim,
                dim,
                3,
                padding=1,
                groups=dim
            ),

            nn.BatchNorm2d(dim),

            nn.Sigmoid()
        )


        # fusion weight

        self.alpha = nn.Parameter(
            torch.tensor(0.5)
        )


    def forward(self,x):

        identity = x


        # ===== LKA branch =====

        attn = self.conv0(x)

        attn = self.conv_spatial(attn)

        attn = self.conv1(attn)


        # ===== defect branch =====

        defect_weight = self.defect(x)


        # defect-aware attention

        attn = attn * (
            1 + self.alpha * defect_weight
        )


        out = identity * attn


        return out