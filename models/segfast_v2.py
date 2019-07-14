# -*- coding: utf-8 -*-

# IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F

# MODULES

# Depthwise Separable Convolution Layer
class DwConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = None):
        super(DwConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if self.padding == None:
            self.padding = int(self.kernel_size/2)
        if self.kernel_size == 3:
            self.layer = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.in_channels,
                                                 kernel_size = (1,3),
                                                 stride = self.stride,
                                                 padding = (0,1),
                                                 groups = self.in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.in_channels,
                                                 kernel_size = (3,1),
                                                 stride = self.stride,
                                                 padding = (1,0),
                                                 groups = self.in_channels),
                                       nn.BatchNorm2d(self.in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.out_channels,
                                                 kernel_size = 1),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.ReLU(inplace=True))
        if self.kernel_size == 5:
            self.layer = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.in_channels,
                                                 kernel_size = (1,3),
                                                 stride = self.stride,
                                                 padding = (0,1),
                                                 groups = self.in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.in_channels,
                                                 kernel_size = (3,1),
                                                 stride = self.stride,
                                                 padding = (1,0),
                                                 groups = self.in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.in_channels,
                                                 kernel_size = (1,3),
                                                 stride = self.stride,
                                                 padding = (0,1),
                                                 groups = self.in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.in_channels,
                                                 kernel_size = (3,1),
                                                 stride = self.stride,
                                                 padding = (1,0),
                                                 groups = self.in_channels),
                                       nn.BatchNorm2d(self.in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.out_channels,
                                                 kernel_size = 1),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.ReLU(inplace=True))
    def forward(self, x):
        return(self.layer(x))
        
# Depthwise Separable Convolution Layer
class DwConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = None):
        super(DwConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if self.padding == None:
            self.padding = int(self.kernel_size/2)
        if self.kernel_size==3:
            self.layer = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.out_channels,
                                                 kernel_size = 1),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels = self.out_channels,
                                                          out_channels = self.out_channels,
                                                          kernel_size = (1,3),
                                                          stride = self.stride,
                                                          padding = (0,1),
                                                          groups = self.out_channels),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels = self.out_channels,
                                                          out_channels = self.out_channels,
                                                          kernel_size = (3,1),
                                                          stride = self.stride,
                                                          padding = (1,0),
                                                          groups = self.out_channels),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.ReLU(inplace=True))
        if self.kernel_size==5:
            self.layer = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                                 out_channels = self.out_channels,
                                                 kernel_size = 1),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels = self.out_channels,
                                                          out_channels = self.out_channels,
                                                          kernel_size = (1,3),
                                                          stride = self.stride,
                                                          padding = (0,1),
                                                          groups = self.out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels = self.out_channels,
                                                          out_channels = self.out_channels,
                                                          kernel_size = (3,1),
                                                          stride = self.stride,
                                                          padding = (1,0),
                                                          groups = self.out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels = self.out_channels,
                                                          out_channels = self.out_channels,
                                                          kernel_size = (1,3),
                                                          stride = self.stride,
                                                          padding = (0,1),
                                                          groups = self.out_channels),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels = self.out_channels,
                                                          out_channels = self.out_channels,
                                                          kernel_size = (3,1),
                                                          stride = self.stride,
                                                          padding = (1,0),
                                                          groups = self.out_channels),
                                       nn.BatchNorm2d(self.out_channels),
                                       nn.ReLU(inplace=True))                                               
    def forward(self, x):
        return(self.layer(x))

# Spark Module
class Spark_Module(nn.Module):
    def __init__(self,
                 in_channels,
                 squeeze_planes,
                 expand_1x1_planes,
                 expand_kxk_planes,
                 kernel_size = 3,
                 padding = None):
        super(Spark_Module, self).__init__()

        self.in_channels = in_channels
        self.squeeze_planes = squeeze_planes
        self.expand_1x1_planes = expand_1x1_planes
        self.expand_kxk_planes = expand_kxk_planes
        self.kernel_size = kernel_size
        self.squeeze_layer = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                                     out_channels = self.squeeze_planes,
                                                     kernel_size = 1),
                                           nn.BatchNorm2d(self.squeeze_planes),
                                           nn.ReLU(inplace=True))
        self.expand_1x1_layer = nn.Sequential(nn.Conv2d(in_channels = self.squeeze_planes,
                                                         out_channels = self.expand_1x1_planes,
                                                         kernel_size = 1),
                                              nn.BatchNorm2d(self.expand_1x1_planes),
                                              nn.ReLU(inplace=True))
        self.expand_kxk_layer = nn.Sequential(DwConv2d(in_channels = self.squeeze_planes,
                                                       out_channels = self.expand_kxk_planes,
                                                       kernel_size = self.kernel_size))

    def forward(self, x):
        x = self.squeeze_layer(x)
        x = torch.cat([self.expand_1x1_layer(x),self.expand_kxk_layer(x)],1)
        return (x)

# Encoder Blocks with spark modules and average pooling
class Spark_Down_Block(nn.Module):      # C x H x W  --> 2C x H/2 x W/2
    def __init__(self,C,k):
        super(Spark_Down_Block, self).__init__()
        self.layer = nn.Sequential(nn.AvgPool2d(2,2),
                                   Spark_Module(C,int(C/2),C,C,k),
                                   Spark_Module(2*C,int(C/2),C,C,k))
    def forward(self, x):
        return(self.layer(x))

# Decoder Blocks with spark modules and upsampling
class Up_Block(nn.Module):        # 2C x H/2 x W/2  --> C x H x W
    def __init__(self,C,k):
        super(Up_Block, self).__init__()
        self.up_layer = nn.Sequential(DwConvTranspose2d(2*C,C,k,2))
        self.concat_layer = Spark_Module(2*C,int(C/2),int(C/2),int(C/2),k)
    def forward(self, x, bridge):
        x = self.up_layer(x)
        x_size = x.size()
        b_size = bridge.size()
        if (x_size!=b_size):
            x = F.pad(x,(0,b_size[3]-x_size[3],0,b_size[2]-x_size[2]))
        x = torch.cat([x,bridge],dim = 1)
        return (self.concat_layer(x))
    
# SegFast Model
class SegFast_V2(nn.Module):
    def __init__(self,C, noc, k):
        super(SegFast_V2, self).__init__()
        self.primary_conv = DwConv2d(3, C, kernel_size = k, stride=2)
        self.down_block_1 = Spark_Down_Block(C, k)
        self.down_block_2 = Spark_Down_Block(2*C, k)
        self.down_block_3 = Spark_Down_Block(4*C, k)
        self.up_block_3 = Up_Block(4*C, k)
        self.up_block_2 = Up_Block(2*C, k)
        self.up_block_1 = Up_Block(C, k)
        self.classifier = nn.Sequential(nn.Conv2d(C,noc,1),
                                        nn.UpsamplingBilinear2d(scale_factor=2))
        
    def forward(self,x):
        x = self.primary_conv(x)            #   C x H/2  x W/2
        d1 = self.down_block_1(x)           # 2*C x H/4  x H/4
        d2 = self.down_block_2(d1)          # 4*C x H/8  x W/8
        d3 = self.down_block_3(d2)          # 8*C x H/16 x W/16
        u3 = self.up_block_3(d3,d2)         # 4*C x H/8  x W/8
        u2 = self.up_block_2(u3,d1)         # 2*C x H/4  x H/4
        u1 = self.up_block_1(u2,x)          #   C x H/2  x W/2
        out = self.classifier(u1)           # noc x H    x W
        return (out)



