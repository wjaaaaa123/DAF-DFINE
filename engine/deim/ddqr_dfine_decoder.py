from engine.deim.dfine_decoder import DFINETransformer


class DDQRDFINETransformer(DFINETransformer):

    def __init__(self,
                 feat_channels,
                 num_classes,
                 eval_spatial_size,
                 ddqr=True,
                 **kwargs):

        super().__init__(
            feat_channels=feat_channels,
            num_classes=num_classes,
            eval_spatial_size=eval_spatial_size,
            **kwargs
        )

        if ddqr:
            from engine.extre_module.custom_nn.attention.ddqr import DDQR

            self.ddqr = DDQR(
                channels=feat_channels,
                hidden_dim=256
            )
        else:
            self.ddqr = None


    def forward(self, feats, targets=None):

        if self.ddqr is not None:
            feats = self.ddqr(feats)

        return super().forward(
            feats,
            targets
        )