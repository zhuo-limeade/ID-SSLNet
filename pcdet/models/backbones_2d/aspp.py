# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, 
                 conv_layer=nn.Conv2d, bias=False, **kwargs):
        super(Conv, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = conv_layer(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)
                        
    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False, conv_layer=conv_layer)

        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False, conv_layer=conv_layer)

        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out

class ASPPNeck(nn.Module):
    def __init__(self, model_cfg, input_channels):

        super(ASPPNeck, self).__init__()
        times = 6
        self.model_cfg = model_cfg
        pre_conv = self.model_cfg.get('PRE_CONV', None)
        out_channels = self.model_cfg.get('OUT_CHANNELS', 256)
        self.num_bev_features = out_channels
        if  pre_conv is not None:
            self.pre_conv_down = nn.Conv2d(input_channels, pre_conv, kernel_size=1, stride=1, bias=False, padding=0)
            self.pre_conv = BasicBlock(pre_conv)
            self.conv1x1 = nn.Conv2d(
                pre_conv, pre_conv, kernel_size=1, stride=1, bias=False, padding=0)
            self.weight = nn.Parameter(torch.randn(pre_conv, pre_conv, 3, 3))
            self.post_conv = ConvBlock(pre_conv * times, out_channels, kernel_size=1, stride=1)
        else:
            self.pre_conv_down=None
            self.pre_conv = BasicBlock(input_channels)
            self.conv1x1 = nn.Conv2d(
                input_channels, input_channels, kernel_size=1, stride=1, bias=False, padding=0)
            self.weight = nn.Parameter(torch.randn(input_channels, input_channels, 3, 3))
            self.post_conv = ConvBlock(input_channels * times, out_channels, kernel_size=1, stride=1)

    def _forward(self, x):
        if self.pre_conv_down:
            x = self.pre_conv_down(x)
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=2, dilation=2)
        branch12 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=4, dilation=4)
        branch18 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=8, dilation=8)
        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        return x
    
    def _forward2(self, x):
        if self.pre_conv_down:
            x = self.pre_conv_down(x)
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=2, dilation=2)
        branch12 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=4, dilation=4)
        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12), dim=1))
        return x

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        out = self._forward(spatial_features)
        # out = self._forward2(x)
        data_dict['spatial_features_2d'] = out
        return data_dict
    
