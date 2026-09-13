import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQR(nn.Module):
    """
    Defect-aware Dynamic Query Refinement

    用于D-FINE Decoder前的多尺度特征增强。
    输入:
        feats: [P3, P4, P5]
    输出:
        refined feats

    主要思想:
        1. 利用P3高分辨率特征提取缺陷响应
        2. 生成动态权重
        3. 对多尺度feature进行自适应增强
    """

    def __init__(
        self,
        channels=[256, 256, 256],
        hidden_dim=256,
        reduction=4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim


        # 多尺度统一映射
        self.proj = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1, bias=False)
            for c in channels
        ])


        # 缺陷响应生成
        self.defect_attention = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim // reduction,
                kernel_size=1
            ),
            nn.BatchNorm2d(
                hidden_dim // reduction
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_dim // reduction,
                hidden_dim,
                kernel_size=1
            ),
            nn.Sigmoid()
        )


        # 动态尺度权重
        self.scale_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                hidden_dim,
                hidden_dim // reduction,
                1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim // reduction,
                3,
                1
            ),
            nn.Softmax(dim=1)
        )


        self.out_proj = nn.ModuleList([
            nn.Conv2d(
                hidden_dim,
                c,
                1,
                bias=False
            )
            for c in channels
        ])


    def forward(self, feats):

        assert len(feats) == 3, \
            "DDQR requires P3,P4,P5 features"


        # channel projection
        x = [
            self.proj[i](feats[i])
            for i in range(3)
        ]


        # P3作为缺陷细节引导
        p3 = x[0]

        attention = self.defect_attention(p3)


        # 缺陷增强
        x = [
            feat * attention
            for feat in x
        ]


        # 多尺度动态权重
        fused = F.interpolate(
            x[1],
            size=x[0].shape[-2:],
            mode="nearest"
        )

        fused = fused + x[0]


        weight = self.scale_weight(fused)


        outputs=[]

        for i in range(3):

            enhanced = x[i] * weight[:,i:i+1]

            outputs.append(
                self.out_proj[i](enhanced)
                + feats[i]
            )


        return outputs