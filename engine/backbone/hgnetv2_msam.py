"""
DEIM-HGNetV2 + MSAM backbone.

HGNetV2-B0 originally outputs P4 and P5 for DEIM-N.
This version first obtains P2, P3, P4 and P5 from HGNetV2,
then uses MSAM_P4 to enhance P4. The final output remains
[enhanced P4, original P5], so the original HybridEncoder can
be reused without changing decoder structure.
"""

from .hgnetv2 import HGNetv2
from ..core import register
from ..extre_module.custom_nn.featurefusion.MSAM import MSAM_P4


__all__ = ["HGNetv2_MSAM"]


@register()
class HGNetv2_MSAM(HGNetv2):
    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=[2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        agg="se",
        local_model_dir="weight/hgnetv2/",
    ):
        if pretrained:
            raise ValueError(
                "HGNetv2_MSAM must use pretrained: False for this experiment."
            )

        # Force HGNetV2 to build all four stage outputs:
        # P2: 64, 160x160
        # P3: 256, 80x80
        # P4: 512, 40x40
        # P5: 1024, 20x20
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

        self.msam_p4 = MSAM_P4(
            [64, 256, 512, 1024],
            512
        )

        # Keep the external output channels same as original DEIM-N:
        # [P4, P5] = [512, 1024]
        self._out_channels = [512, 1024]
        self._out_strides = [16, 32]

        print("[HGNetv2_MSAM] MSAM_P4 enabled. Output: [enhanced P4, P5].")

    def forward(self, x):
        x = self.stem(x)

        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        p2, p3, p4, p5 = feats

        p4 = self.msam_p4([p2, p3, p4, p5])

        return [p4, p5]