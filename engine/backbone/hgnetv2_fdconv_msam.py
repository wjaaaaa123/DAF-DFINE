
from .hgnetv2_fdconv import HGNetv2_FDConv
from ..core import register
from ..extre_module.custom_nn.featurefusion.MSAM import MSAM_P4


__all__ = ["HGNetv2_FDConv_MSAM"]


@register()
class HGNetv2_FDConv_MSAM(HGNetv2_FDConv):
    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=[2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=False,
        agg="se",
        local_model_dir="weight/hgnetv2/",
    ):
        # 先创建FDConv版本，并让主干保留四个尺度输出
        super().__init__(
            name=name,
            use_lab=use_lab,
            return_idx=[0, 1, 2, 3],
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=False,
            agg=agg,
            local_model_dir=local_model_dir,
        )

        # HGNetV2-B0四个阶段通道
        self.msam_p4 = MSAM_P4(
            [64, 256, 512, 1024],
            512
        )

        self._out_channels = [512, 1024]
        self._out_strides = [16, 32]

        print(
            "[HGNetv2_FDConv_MSAM] "
            "FDConv stage2 + MSAM_P4 enabled."
        )

    def forward(self, x):
        x = self.stem(x)

        feats = []

        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        p2, p3, p4, p5 = feats

        enhanced_p4 = self.msam_p4(
            [p2, p3, p4, p5]
        )

        return [enhanced_p4, p5]