"""
DEIM-HGNetV2 + FDConv backbone.

Only the standard 3x3 convolution layers inside stage2 HG_Block
are replaced by FDConv. Stem, downsampling, LightConv and aggregation
layers remain unchanged.
"""

import torch.nn as nn

from .hgnetv2 import HGNetv2, LearnableAffineBlock
from ..core import register
from ..extre_module.custom_nn.conv_module.FDConv import FDConv


__all__ = ["HGNetv2_FDConv"]


class FDConvBNAct(nn.Module):
    """FDConv + BatchNorm + ReLU + optional LAB."""

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=3,
        stride=1,
        groups=1,
        use_act=True,
        use_lab=False,
    ):
        super().__init__()

        self.conv = FDConv(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=False,
            use_fdconv_if_c_gt=16,
            use_fdconv_if_k_in=[3],
            use_fbm_if_k_in=[3],
            kernel_num=16,
        )

        self.bn = nn.BatchNorm2d(out_chs)

        if use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        if use_act and use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


@register()
class HGNetv2_FDConv(HGNetv2):
    """
    HGNetV2 backbone with FDConv in stage2.

    For HGNetV2-B0, stage2 contains one HG_Block with three
    standard 3x3 convolution layers. These three layers are
    replaced by FDConv.
    """

    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        agg="se",
        local_model_dir="weight/hgnetv2/",
    ):
        if pretrained:
            raise ValueError(
                "HGNetv2_FDConv must use pretrained: False for this experiment."
            )

        super().__init__(
            name=name,
            use_lab=use_lab,
            return_idx=return_idx,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=False,
            agg=agg,
            local_model_dir=local_model_dir,
        )

        stage_name = "stage2"
        stage_config = self.arch_configs[name]["stage_config"]
        stage_names = list(stage_config.keys())
        stage_index = stage_names.index(stage_name)

        (
            in_channels,
            mid_channels,
            out_channels,
            block_num,
            downsample,
            light_block,
            kernel_size,
            layer_num,
        ) = stage_config[stage_name]

        if light_block:
            raise ValueError(
                f"{stage_name} uses LightConv and cannot use this replacement."
            )

        replaced_count = 0

        for block_index, block in enumerate(
            self.stages[stage_index].blocks
        ):
            block_in_channels = (
                in_channels if block_index == 0 else out_channels
            )

            new_layers = nn.ModuleList()

            for layer_index in range(layer_num):
                layer_in_channels = (
                    block_in_channels
                    if layer_index == 0
                    else mid_channels
                )

                new_layers.append(
                    FDConvBNAct(
                        in_chs=layer_in_channels,
                        out_chs=mid_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=1,
                        use_act=True,
                        use_lab=use_lab,
                    )
                )

                replaced_count += 1

            block.layers = new_layers

        print(
            f"[HGNetv2_FDConv] {stage_name}: "
            f"{replaced_count} convolution layers replaced."
        )