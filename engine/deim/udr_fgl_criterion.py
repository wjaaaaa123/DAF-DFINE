"""
UDR-FGL for the magic-modified D-FINE/DEIM project.

Recommended location:
    engine/deim/udr_fgl_criterion.py

UDR-FGL = Uncertainty-aware Distribution Refinement Fine-grained
Localization Loss.

Implementation principle:
1) call the project's original DEIMCriterion.loss_local() first, preserving
   its DDF/self-distillation behavior and all existing target-cache logic;
2) recompute only loss_fgl using:
       normalized distribution entropy U
       localization quality Q (matched IoU)
       A = Q^alpha * (1 + lambda_u * U^gamma)
3) add a small uncertainty-quality consistency term:
       confidence of distribution ~= localization quality
   where distribution confidence is (1 - normalized entropy).

This avoids edge-specific top/bottom/left/right weighting and therefore keeps
the innovation centered on distribution uncertainty rather than edge geometry.
"""

import math
import torch
import torch.nn.functional as F

from ..core import register
from .deim_criterion import DEIMCriterion
from .box_ops import box_cxcywh_to_xyxy, box_iou


@register()
class UDRDEIMCriterion(DEIMCriterion):
    def __init__(
        self,
        *args,
        udr_lambda_u=0.50,
        udr_gamma=1.0,
        udr_quality_power=1.0,
        udr_calib_weight=0.05,
        udr_eps=1e-8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.udr_lambda_u = float(udr_lambda_u)
        self.udr_gamma = float(udr_gamma)
        self.udr_quality_power = float(udr_quality_power)
        self.udr_calib_weight = float(udr_calib_weight)
        self.udr_eps = float(udr_eps)

    def _normalized_entropy(self, pred_corners):
        prob = F.softmax(pred_corners.float(), dim=-1)
        prob = prob.clamp_min(self.udr_eps)
        entropy = -(prob * prob.log()).sum(dim=-1)
        entropy = entropy / math.log(self.reg_max + 1)
        return entropy.clamp(0.0, 1.0)

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        # Preserve the project's original local-loss path, especially DDF.
        losses = super().loss_local(
            outputs, targets, indices, num_boxes, T=T
        )

        if "pred_corners" not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)

        # If a batch has no matched positive sample, keep the original result.
        if idx[0].numel() == 0:
            return losses

        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        pred_corners = outputs["pred_corners"][idx].reshape(
            -1, self.reg_max + 1
        )

        # super().loss_local() has already populated these caches.
        cached_targets = (
            self.fgl_targets_dn if "is_dn" in outputs
            else self.fgl_targets
        )
        if cached_targets is None:
            return losses

        target_corners, weight_right, weight_left = cached_targets

        with torch.no_grad():
            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                    box_cxcywh_to_xyxy(target_boxes)
                )[0]
            ).clamp(0.0, 1.0)

            # One localization quality is shared by the four box-edge distributions.
            quality = ious[:, None].expand(-1, 4).reshape(-1)

        entropy = self._normalized_entropy(pred_corners)

        # Adaptive FGL weight:
        # high-quality + uncertain samples receive more localization attention.
        adaptive_weight = (
            quality.pow(self.udr_quality_power)
            * (
                1.0
                + self.udr_lambda_u
                * entropy.detach().pow(self.udr_gamma)
            )
        )

        loss_fgl_udr = self.unimodal_distribution_focal_loss(
            pred_corners,
            target_corners,
            weight_right,
            weight_left,
            adaptive_weight,
            avg_factor=num_boxes
        )

        # Distribution confidence (1-U) should be consistent with localization quality.
        # This term is intentionally small; it regularizes probability sharpness
        # without replacing D-FINE's original FGL objective.
        dist_confidence = 1.0 - entropy
        target_quality = quality.detach()

        loss_uqc = F.smooth_l1_loss(
            dist_confidence,
            target_quality,
            reduction="sum"
        ) / (num_boxes * 4.0)

        losses["loss_fgl"] = (
            loss_fgl_udr
            + self.udr_calib_weight * loss_uqc
        )

        # loss_ddf (if present) remains exactly the one produced by the parent class.
        return losses
