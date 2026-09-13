"""
LSGI: Lightweight Structure-Guided Interaction Block
Intended to replace the P5/32 TransformerEncoderBlock in the D-FINE hybrid encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFFN(nn.Module):
    def __init__(self,dim,ratio=2.0):
        super().__init__(); hidden=int(dim*ratio)
        self.net=nn.Sequential(nn.Conv2d(dim,hidden,1,bias=False),nn.BatchNorm2d(hidden),nn.SiLU(inplace=True),nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden,bias=False),nn.BatchNorm2d(hidden),nn.SiLU(inplace=True),nn.Conv2d(hidden,dim,1,bias=False),nn.BatchNorm2d(dim))
    def forward(self,x): return self.net(x)

class StructureGuide(nn.Module):
    def __init__(self):
        super().__init__(); self.refine=nn.Sequential(nn.Conv2d(1,1,3,padding=1,bias=True),nn.Sigmoid())
    def forward(self,x):
        smooth=F.avg_pool2d(x,3,1,1); contrast=torch.abs(x-smooth).mean(1,keepdim=True); return self.refine(contrast)

class LSGI(nn.Module):
    def __init__(self,dim,qk_dim=16,n_div=4):
        super().__init__(); assert dim % n_div == 0
        self.dim=dim; self.qk_dim=qk_dim; self.pdim=dim//n_div; self.scale=qk_dim**-0.5
        self.pre_norm=nn.GroupNorm(1,dim); self.local_mix=nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=False)
        self.in_proj=nn.Conv2d(dim,2*qk_dim+dim,1,bias=False); self.structure=StructureGuide(); self.out_proj=nn.Conv2d(dim,dim,1,bias=False)
    def forward(self,x):
        z=self.local_mix(self.pre_norm(x)); q,k,v,u=self.in_proj(z).split([self.qk_dim,self.qk_dim,self.pdim,self.dim-self.pdim],dim=1)
        s=self.structure(z); v=v*(1+s); u=u*(1+s)
        q=q.flatten(2); k=k.flatten(2); v=v.flatten(2)
        attn=(q.transpose(-2,-1)@k)*self.scale; attn=attn.softmax(dim=-1)
        B,_,H,W=u.shape; global_feat=(v@attn.transpose(-2,-1)).reshape(B,self.pdim,H,W)
        return self.out_proj(torch.cat([global_feat,u],dim=1))

class LSGIBlock(nn.Module):
    def __init__(self,dim,qk_dim=16,n_div=4,ffn_ratio=2.0):
        super().__init__(); self.interaction=LSGI(dim,qk_dim,n_div); self.ffn_norm=nn.GroupNorm(1,dim); self.ffn=ConvFFN(dim,ffn_ratio)
    def forward(self,x):
        x=x+self.interaction(x); x=x+self.ffn(self.ffn_norm(x)); return x

if __name__=='__main__':
    x=torch.randn(1,256,20,20); m=LSGIBlock(256); print(m(x).shape)
