# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class UpBlock_mobile(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
            
        def tconv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(inp, inp, 2, stride, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
            
        super(UpBlock_mobile, self).__init__()
        #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size= 2, stride=2)
        self.up = tconv_dw(in_size, out_size, 2)
        self.conv = conv_dw(in_size, out_size,1)

    def forward(self, x, bridge):
        #print("x",x.size())
        #print("bridge",bridge.size())
        up = self.up(x)
        #print("up",up.size())
        
        if bridge.size()[2:] != up.size()[2:]:
            crop1 = F.upsample(bridge,size=up.size()[2:],mode='bilinear', align_corners = True)
        else:
            crop1 = bridge
            
        out = torch.cat([up, crop1], 1)
        out = self.conv(out)
        return out
    
class SegFast_Mobile(nn.Module):
    def __init__(self,nc):
        super(SegFast_Mobile, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.num_classes = nc
        self.a = conv_bn(  3,  32, 2)
        self.b = conv_dw( 32,  64, 1)
        self.c = conv_dw( 64, 128, 2)
        self.d = conv_dw(128, 128, 1)
        self.e = conv_dw(128, 256, 2)
        self.f = conv_dw(256, 256, 1)
        self.g = conv_dw(256, 512, 2)
        self.h = conv_dw(512, 512, 1)
        self.i = conv_dw(512, 512, 1)
        self.j = conv_dw(512, 512, 1)
        self.k = conv_dw(512, 512, 1)
        self.l = conv_dw(512, 512, 1)
        self.m = conv_dw(512, 1024, 2)
        self.n = conv_dw(1024, 1024, 1)
        
        self.up3 = UpBlock_mobile(1024,512)
        self.up2 = UpBlock_mobile(512,256)
        self.up1 = UpBlock_mobile(256,128)
        self.up0 = UpBlock_mobile(128,64)
        self.last = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        xa = self.a(x)
        xb = self.b(xa)
        xc = self.c(xb)
        xd = self.d(xc)
        xe = self.e(xd)
        xf = self.f(xe)
        xg = self.g(xf)
        xh = self.h(xg)
        xi = self.i(xh)
        xj = self.j(xi)
        xk = self.k(xj)
        xl = self.l(xk)
        xm = self.m(xl)
        xn = self.n(xm)
        up3 = self.up3(xn,xl)
        up2 = self.up2(up3,xf)
        up1 = self.up1(up2,xd)
        up0 = self.up0(up1,xb)
        return F.upsample(self.last(up0),size=x.size()[2:],mode='bilinear', align_corners = True)