"""
DHIC: Detail-aware High-order Interaction Convolution
Prototype for D-FINE-S industrial surface-defect detection.
"""
import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,d=1,act=True):
        super().__init__(); p = d*(k-1)//2 if p is None else p
        self.conv=nn.Conv2d(c1,c2,k,s,p,dilation=d,groups=g,bias=False)
        self.bn=nn.BatchNorm2d(c2)
        self.act=nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class DHIC(nn.Module):
    def __init__(self,c1,c2,expansion=1.0,context_kernel=5,dilation=2):
        super().__init__(); hidden=max(int(c1*expansion),16)
        self.proj_in=ConvBNAct(c1,hidden,1,1)
        self.fine=ConvBNAct(hidden,hidden,3,1,g=hidden)
        self.context=ConvBNAct(hidden,hidden,context_kernel,1,g=hidden,d=dilation)
        self.response_gate=nn.Sequential(nn.Conv2d(hidden,hidden,1,bias=True),nn.Sigmoid())
        self.beta=nn.Parameter(torch.tensor(0.10))
        self.consolidate=nn.Sequential(ConvBNAct(hidden,hidden,3,1,g=hidden),ConvBNAct(hidden,c2,1,1,act=False))
        self.out_act=nn.SiLU(inplace=True)
    def forward(self,x):
        h=self.proj_in(x); f_detail=self.fine(h); f_context=self.context(h)
        gate=self.response_gate(torch.abs(f_detail-f_context))
        first_order=0.5*(f_detail+f_context)
        second_order=f_detail*f_context
        y=h+first_order+self.beta*gate*second_order
        return self.out_act(self.consolidate(y))

if __name__=='__main__':
    x=torch.randn(1,256,80,80); m=DHIC(256,256); print(m(x).shape)
