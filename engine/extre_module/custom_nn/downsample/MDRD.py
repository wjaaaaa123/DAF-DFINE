"""
MDRD: Morphology-Driven Routing Downsampling
Prototype for replacing the first SCDown (P3/8 -> P4/16) in D-FINE-S.
This is an independently written prototype, not a verbatim rewrite of RouterLAWDS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,act=True):
        super().__init__()
        if p is None:
            p=tuple((kk-1)//2 for kk in k) if isinstance(k,tuple) else (k-1)//2
        self.conv=nn.Conv2d(c1,c2,k,s,p,groups=g,bias=False)
        self.bn=nn.BatchNorm2d(c2); self.act=nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class MDRD(nn.Module):
    def __init__(self,c1,c2,router_hidden=8):
        super().__init__(); self.pre=ConvBNAct(c1,c2,1,1)
        self.detail=ConvBNAct(c2,c2,3,2,g=c2)
        self.dir_h=ConvBNAct(c2,c2,(1,5),2,g=c2); self.dir_v=ConvBNAct(c2,c2,(5,1),2,g=c2)
        self.context=ConvBNAct(c2,c2,5,2,g=c2)
        self.router=nn.Sequential(nn.Conv2d(3,router_hidden,1),nn.SiLU(inplace=True),nn.Conv2d(router_hidden,3,1))
        self.post=ConvBNAct(c2,c2,1,1)
    @staticmethod
    def morphology_descriptor(x):
        gray=x.mean(1,keepdim=True); smooth=F.avg_pool2d(gray,3,1,1); local=torch.abs(gray-smooth)
        gx=torch.abs(gray[:,:,:,1:]-gray[:,:,:,:-1]); gx=F.pad(gx,(0,1,0,0))
        gy=torch.abs(gray[:,:,1:,:]-gray[:,:,:-1,:]); gy=F.pad(gy,(0,0,0,1))
        return torch.cat([local,gx,gy],1)
    def forward(self,x):
        h=self.pre(x); f_detail=self.detail(h); f_dir=0.5*(self.dir_h(h)+self.dir_v(h)); f_context=self.context(h)
        desc=F.avg_pool2d(self.morphology_descriptor(x),2,2); route=torch.softmax(self.router(desc),dim=1)
        y=route[:,0:1]*f_detail+route[:,1:2]*f_dir+route[:,2:3]*f_context
        return self.post(y)

if __name__=='__main__':
    x=torch.randn(1,256,80,80); m=MDRD(256,256); print(m(x).shape)
