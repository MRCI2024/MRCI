from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import functools
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import cv2
from arch import Bottleneck,ASPP,Resizer,stageblock,LastLayer,MultibranchFusion,Head
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
from functools import partial

class MultibranchNet(nn.Module):
    def __init__(self):
        super(MultibranchNet, self).__init__() 
        num_channels =64
        num_blocks = 4 
        self.reszier_128 = Resizer(in_chs=4, out_size=(64,64), n_filters = 16, n_res_blocks = 2)
        self.reszier_96 = Resizer(in_chs=4, out_size=(64,64), n_filters = 16, n_res_blocks = 2)
        self.reszier_64 = Resizer(in_chs=4, out_size=(64,64), n_filters = 16, n_res_blocks = 2)
        self.head1 = Head(num_channels, num_blocks) 
        self.head2 = Head(num_channels, num_blocks)
        self.head3 = Head(num_channels, num_blocks) 

        Fw = num_channels*2 
        self.embed_cs = nn.Linear(34, Fw)
        self.embed_pos_128 = nn.Linear(72*2, Fw)
        self.embed_pos_96 = nn.Linear(72*2, Fw)
        self.embed_pos_64 = nn.Linear(72*2, Fw)
        self.pos_weight = nn.Parameter(torch.tensor([0.1]))
        self.cs_weight = nn.Parameter(torch.tensor([0.1]))
        
        stage1_out_channel = num_channels*4
        
        dilation_rate = [6,12,18] #deeplab 源码中的dilation_rate
        self.ASPP_model=self._make_ASPP_layer(stage1_out_channel,stage1_out_channel,dilation_rate)
        self.stage1, pre_stage_channels = self._make_fuse_layer(stage1_out_channel) #pre_stage_channels=num_channels，没啥用 #_make_stage用于不同分支融合
        
        #stage2
        num_channels =64
        self.transition1 = self._make_stage_layer(
            [stage1_out_channel, stage1_out_channel, stage1_out_channel], [num_channels,num_channels,num_channels]) 
        self.stage2, pre_stage_channels = self._make_fuse_layer(num_channels) #pre_stage_channels=num_channels，没啥用 #_make_stage用于不同分支融合

        #stage3
        num_channels = 64 
        self.transition2 = self._make_stage_layer(
            [pre_stage_channels,pre_stage_channels,pre_stage_channels],  [num_channels,num_channels,num_channels]) # #主干网络输入pre_stage_channels，输出通道num_channels
        self.stage3, pre_stage_channels = self._make_fuse_layer(num_channels) # （不同分支融合）

        #stage4
        num_channels = 96
        self.transition3 = self._make_stage_layer(
            [pre_stage_channels,pre_stage_channels,pre_stage_channels], [num_channels,num_channels,num_channels])   #主网络 输入：pre_stage_channels，输出num_channels
        self.stage4, pre_stage_channels = self._make_fuse_layer(num_channels) #分支进行裁剪融合
        
        pre_stage_channels =[pre_stage_channels,pre_stage_channels,pre_stage_channels]
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        
        self.last_layer = LastLayer(last_inp_channels)
        last_inp_channels = int(last_inp_channels/len(pre_stage_channels))
        self.last_layer1 = LastLayer(last_inp_channels)
        self.last_layer2 = LastLayer(last_inp_channels)
        self.last_layer3 = LastLayer(last_inp_channels)


    def _make_stage_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            transition_layers.append(stageblock(num_channels_pre_layer[i],num_channels_cur_layer[i]))
        return nn.ModuleList(transition_layers)

    def _make_fuse_layer(self,num_inchannels,method="cat"):
        return MultibranchFusion(num_inchannels,method=method),num_inchannels
    
    def _make_ASPP_layer(self,num_channel_pre_layer,num_channel_cur_layer,dilation_rate):
        num_branches_cur = 3
        modules=[]
        for i in range(num_branches_cur):
            modules.append(ASPP(num_channel_pre_layer,num_channel_cur_layer,dilation_rate))
        return nn.ModuleList(modules)

    def forward(self, x_list,pos_mess):
        x_512, x_256, x_128 = x_list
        x1 = self.reszier_128(x_512)
        x2 = self.reszier_96(x_256)
        x3 = self.reszier_64(x_128)
        x1 = self.head1(x1)
        x2 = self.head2(x2)
        x3 = self.head3(x3)

        pos128_embed = self.embed_pos_128(pos_mess[0])
        pos96_embed = self.embed_pos_96(pos_mess[1]) 
        pos64_embed = self.embed_pos_64(pos_mess[2]) 
        cs_embed = self.embed_cs(pos_mess[3])
        pos128_embed = pos128_embed.unsqueeze(2).unsqueeze(3)
        pos96_embed = pos96_embed.unsqueeze(2).unsqueeze(3)        
        pos64_embed = pos64_embed.unsqueeze(2).unsqueeze(3)
        cs_embed = cs_embed.unsqueeze(2).unsqueeze(3)        
        x1 = x1 + self.pos_weight*pos128_embed + self.cs_weight*cs_embed
        x2 = x2 + self.pos_weight*pos96_embed + self.cs_weight*cs_embed
        x3 = x3 + self.pos_weight*pos96_embed + self.cs_weight*cs_embed
        x_list = [x1, x2, x3]
        z_list=[]
        for i in range(3):
            if self.ASPP_model[i] is not None:
                z_list.append(self.ASPP_model[i](x_list[i]))
            else:
                z_list.append(x_list[i])
        y_list = self.stage1(z_list)
        #stage2
        x_list = []
        for i in range(3):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](y_list[i]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage2(x_list)
        #stage3
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[i]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        #stage4
        x_list = []
        for i in range(3):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[i]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        y0 = F.upsample(x[0], size=(x_512.size(2), x_512.size(3)), mode='bilinear') if x[0].size(2) != x_512.size(2) or x[0].size(3) != x_512.size(3) else x[0]
        y1 = F.upsample(x[1], size=(x_256.size(2), x_256.size(3)), mode='bilinear') if x[1].size(2) != x_256.size(2) or x[1].size(3) != x_256.size(3) else x[1]
        y2 = F.upsample(x[2], size=(x_128.size(2), x_128.size(3)), mode='bilinear') if x[2].size(2) != x_128.size(2) or x[2].size(3) != x_128.size(3) else x[2]
        m0 = y0[:, :, int(x_512.size(2)/2-x_128.size(2)/2):int(x_512.size(2)/2+x_128.size(2)/2), int(x_512.size(3)/2-x_128.size(3)/2):int(x_512.size(3)/2+x_128.size(3)/2)]
        m1 = y1[:, :, int(x_256.size(2)/2-x_128.size(2)/2):int(x_256.size(2)/2+x_128.size(2)/2), int(x_256.size(3)/2-x_128.size(3)/2):int(x_256.size(3)/2+x_128.size(3)/2)]
        m2 = y2
        y = torch.cat([m0, m1, y2], 1)
        y = self.last_layer(y)
        y_0 = self.last_layer1(y0)
        y_1 = self.last_layer2(y1)
        y_2 = self.last_layer3(y2)
        return y_0,y_1,y_2,y,m0,m1,m2