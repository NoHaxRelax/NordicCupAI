import torch
from torch import nn
from torch.nn import functional as F


def block(a,b,stride=1):
 return nn.Sequential(nn.Conv2d(a,b,3,stride,1,bias=False),nn.BatchNorm2d(b),nn.SiLU(),nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.SiLU())


class AssetNet(nn.Module):
 def __init__(self,classes=16):
  super().__init__();self.stem=block(3,32,2);self.s4=block(32,64,2);self.s8=block(64,128,2);self.s16=block(128,192,2)
  self.up8=block(192+128,128);self.up4=block(128+64,96)
  self.heat=nn.Conv2d(96,classes,1);self.box=nn.Conv2d(96,4,1);nn.init.constant_(self.heat.bias,-2.19)
 def forward(self,x):
  x=self.stem(x);s4=self.s4(x);s8=self.s8(s4);s16=self.s16(s8)
  z=self.up8(torch.cat([F.interpolate(s16,size=s8.shape[-2:],mode='bilinear',align_corners=False),s8],1))
  z=self.up4(torch.cat([F.interpolate(z,size=s4.shape[-2:],mode='bilinear',align_corners=False),s4],1))
  return self.heat(z),self.box(z)


def loss(outputs,target,box,mask):
 logits,pred=outputs;pred=pred.float();prob=logits.float().sigmoid().clamp(1e-5,1-1e-5)
 pos=target.eq(1);neg=(target>=0)&(target<1);weights=(1-target.clamp(0,1))**4
 focal=-(torch.log(prob)*(1-prob)**2*pos+torch.log(1-prob)*prob**2*weights*neg).sum()/pos.sum().clamp_min(1)
 regression=F.smooth_l1_loss(pred*mask,box*mask,reduction='sum')/mask.sum().clamp_min(1)
 return focal+regression*2, focal.detach(),regression.detach()
