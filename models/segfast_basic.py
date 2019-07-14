# IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F

# MODULES

# Basic Separable Convolution Layer (For Ablation Study)
class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = None):
        super(BasicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if self.padding == None:
            self.padding = int(self.kernel_size/2)
        self.layer = nn.Sequential(nn.Conv2d(in_channels = self.in_channels,
                                             out_channels = self.out_channels,
                                             kernel_size = self.kernel_size,
                                             stride = self.stride,
                                             padding = self.padding),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU(inplace=True))
    def forward(self, x):
        return(self.layer(x))
        
# Basic Separable Convolution Layer (For Ablation Study)
class BasicConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = None):
        super(BasicConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if self.padding == None:
            self.padding = int(self.kernel_size/2)
        self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels = self.in_channels,
                                                      out_channels = self.out_channels,
                                                      kernel_size = self.kernel_size,
                                                      stride = self.stride,
                                                      padding = self.padding),
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
        self.expand_kxk_layer = nn.Sequential(BasicConv2d(in_channels = self.squeeze_planes,
                                                       out_channels = self.expand_kxk_planes,
                                                       kernel_size = self.kernel_size))

    def forward(self, x):
        x = self.squeeze_layer(x)
        x = torch.cat([self.expand_1x1_layer(x),self.expand_kxk_layer(x)],1)
        return (x)

# Encoder Blocks with spark modules and average pooling
class Spark_Down_Block(nn.Module):      # C x H x W  --> 2C x H/2 x W/2
    def __init__(self,C):
        super(Spark_Down_Block, self).__init__()
        self.layer = nn.Sequential(nn.AvgPool2d(2,2),
                                   Spark_Module(C,int(C/2),C,C),
                                   Spark_Module(2*C,int(C/2),C,C))
    def forward(self, x):
        return(self.layer(x))

# Decoder Blocks with spark modules and upsampling
class Up_Block(nn.Module):        # 2C x H/2 x W/2  --> C x H x W
    def __init__(self,C):
        super(Up_Block, self).__init__()
        self.up_layer = nn.Sequential(BasicConvTranspose2d(2*C,C,3,2,1))
        self.concat_layer = Spark_Module(2*C,int(C/2),int(C/2),int(C/2))
    def forward(self, x, bridge):
        x = self.up_layer(x)
        x_size = x.size()
        b_size = bridge.size()
        if (x_size!=b_size):
            x = F.pad(x,(0,b_size[3]-x_size[3],0,b_size[2]-x_size[2]))
        x = torch.cat([x,bridge],dim = 1)
        return (self.concat_layer(x))
    
# SegFast Model
class SegFast_Basic(nn.Module):
    def __init__(self,C, noc):
        super(SegFast_Basic, self).__init__()
        self.primary_conv = BasicConv2d(3, C, kernel_size=3, stride=2)
        self.down_block_1 = Spark_Down_Block(C)
        self.down_block_2 = Spark_Down_Block(2*C)
        self.down_block_3 = Spark_Down_Block(4*C)
        self.up_block_3 = Up_Block(4*C)
        self.up_block_2 = Up_Block(2*C)
        self.up_block_1 = Up_Block(C)
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


