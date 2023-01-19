import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from functools import partial
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

def conv3x3(in_chs, out_chs = 16):
    return nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=1)
def conv1x1(in_chs, out_chs = 16):
    return nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0)
def conv7x7(in_chs, out_chs = 16):
    return nn.Conv2d(in_chs, out_chs, kernel_size=7, stride=1, padding=3)

class ResBlock(nn.Module):
    def __init__(self, in_chs,out_chs = 16):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_chs, out_chs),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(0.2),
            conv3x3(out_chs, out_chs),
            nn.BatchNorm2d(out_chs)
        )
    def forward(self, x):
        identity = x
        out = self.layers(x)
        out += identity
        return out

class Resizer(nn.Module):
    def __init__(self, in_chs, out_size=(352,352), n_filters = 16, n_res_blocks = 2, mode ='bilinear'):
        super(Resizer, self).__init__()
        self.interpolate_layer = partial(F.interpolate, size=out_size, mode=mode,
            align_corners=True)
        self.conv_layers = nn.Sequential(
            conv7x7(in_chs, n_filters),
            nn.LeakyReLU(0.2),
            conv1x1(n_filters, n_filters),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(n_filters)
        )
        self.residual_layers = nn.Sequential()
        for i in range(n_res_blocks):
            self.residual_layers.add_module(f'res{i}', ResBlock(n_filters, n_filters))
        self.residual_layers.add_module('conv3x3_re', conv3x3(n_filters, n_filters))
        self.residual_layers.add_module('bn', nn.BatchNorm2d(n_filters))
        self.final_conv = conv7x7(n_filters, in_chs)

    def forward(self, x):
        identity = self.interpolate_layer(x)
        conv_out = self.conv_layers(x)
        conv_out = self.interpolate_layer(conv_out)
        conv_out_identity = conv_out
        res_out = self.residual_layers(conv_out)
        res_out += conv_out_identity
        out = self.final_conv(res_out)
        out += identity
        return out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels,out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        #out += residual
        out1 = out + residual
        out1 = self.relu(out1)
        return out1


class MultibranchFusion(nn.Module):
    def __init__(self, num_inchannels, method ="sum"):
        super(MultibranchFusion, self).__init__()
        self.method = method
        self.conv2d1 = nn.Sequential(
            conv3x3(num_inchannels[0]+num_inchannels[1], num_inchannels[1], stride=1),
            BatchNorm2d(num_inchannels[1], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv2d2 = nn.Sequential(
            conv3x3(num_inchannels[1]+num_inchannels[2], num_inchannels[2], stride=1),
            BatchNorm2d(num_inchannels[2], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_fuse = []
        x_fuse.append(x[0])
        if self.method == "sum":
            x2 = x[1][:, :, int(x[1].size(2)/2-x[2].size(2)/2):int(x[1].size(2)/2+x[2].size(2)/2), int(x[1].size(3)/2-x[2].size(3)/2):int(x[1].size(3)/2+x[2].size(3)/2)]+x[2]
            x1 = x[0][:, :, int(x[0].size(2)/2-x[1].size(2)/2):int(x[0].size(2)/2+x[1].size(2)/2), int(x[0].size(3)/2-x[1].size(3)/2):int(x[0].size(3)/2+x[1].size(3)/2)]+x[1]
        elif self.method == "cat":
            x2 = torch.cat([x[1][:, :, int(x[1].size(2)/2-x[2].size(2)/2):int(x[1].size(2)/2+x[2].size(2)/2), int(x[1].size(3)/2-x[2].size(3)/2):int(x[1].size(3)/2+x[2].size(3)/2)],x[2]],dim=1)
            x1 = torch.cat([x[0][:, :, int(x[0].size(2)/2-x[1].size(2)/2):int(x[0].size(2)/2+x[1].size(2)/2), int(x[0].size(3)/2-x[1].size(3)/2):int(x[0].size(3)/2+x[1].size(3)/2)],x[1]],dim=1)
            x2 = self.conv2d2(x2)
            x1 = self.conv2d1(x1)
        x_fuse.append(x1)
        x_fuse.append(x2)
        return x_fuse

class Head(nn.Module):
    def __init__(self, num_channels, num_blocks):
        super(Head, self).__init__() 
        self.block = Bottleneck
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU()
        self.layer = self._make_layer(self.block, 64, self.num_channels, self.num_blocks)
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion #下个block的输入通道=第一个block的输出 planes * block.
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))# 后面每个block输入==输出：planes*expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.layer(x)
        return x

class LastLayer(nn.Module):
    def __init__(self, last_inp_channels):
        super(LastLayer, self).__init__() 
        self.last_inp_channels = last_inp_channels
        self.kernel_size = 1
        self.conv1 = nn.Conv2d(in_channels=self.last_inp_channels, out_channels=self.last_inp_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = BatchNorm2d(self.last_inp_channels, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.last_inp_channels, out_channels=1,kernel_size=self.kernel_size, stride=1, padding=1 if self.kernel_size == 3 else 0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

# stage
class stageblock(nn.Module):
    def __init__(self, num_pre_layer, num_cur_layer):
        super(stageblock, self).__init__() 
        self.block = nn.Sequential(
                    nn.Conv2d(num_pre_layer,
                              num_cur_layer,
                              3,
                              1,
                              1,
                              bias=False),
                    BatchNorm2d(
                        num_cur_layer, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_cur_layer,
                              num_cur_layer,
                              3,
                              1,
                              1,
                              bias=False),
                    BatchNorm2d(
                        num_cur_layer, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True))
    def forward(self,x):
        return self.block(x)