"""
QAD-FGL: Quality-Adaptive Distribution Fine-Grained Localization Loss
Minimal patch for a D-FINE-style criterion.
"""
import math
import torch
import torch.nn.functional as F

def normalized_distribution_entropy(pred_corners,eps=1e-8):
    prob=F.softmax(pred_corners,dim=-1)
    ent=-(prob*torch.log(prob.clamp_min(eps))).sum(-1)
    return ent/max(math.log(pred_corners.shape[-1]),eps)

def qad_fgl_weights(pred_corners,ious,alpha_q=1.0,lambda_u=0.50,gamma_u=1.0):
    entropy=normalized_distribution_entropy(pred_corners).detach()
    quality=ious.reshape(-1,1).repeat(1,4).reshape(-1).clamp(0.,1.)
    return quality.pow(alpha_q)*(1.0+lambda_u*entropy.pow(gamma_u))

def unimodal_distribution_focal_loss_qad(pred,label,weight_right,weight_left,ious,avg_factor,alpha_q=1.0,lambda_u=0.50,gamma_u=1.0):
    dis_left=label.long(); dis_right=dis_left+1
    loss=F.cross_entropy(pred,dis_left,reduction='none')*weight_left.reshape(-1)+F.cross_entropy(pred,dis_right,reduction='none')*weight_right.reshape(-1)
    loss=loss*qad_fgl_weights(pred,ious,alpha_q,lambda_u,gamma_u).float()
    return loss.sum()/avg_factor

# Minimal patch in official D-FINE loss_local():
# Replace the original IoU-only weight_targets + self.unimodal_distribution_focal_loss(...)
# with:
# losses['loss_fgl'] = unimodal_distribution_focal_loss_qad(
#     pred_corners, target_corners, weight_right, weight_left,
#     ious=ious.detach(), avg_factor=num_boxes,
#     alpha_q=1.0, lambda_u=0.50, gamma_u=1.0)
