import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        #self.dropout = nn.Dropout3d(p=0.3, inplace=False)

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

        out += residual
        out = self.relu(out)

        return out

class BottleneckAG(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, downsample=None):
        super(BottleneckAG, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = nn.Conv3d(inplanes, int(outplanes/4), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(int(outplanes/4))
        self.conv2 = nn.Conv3d(
            int(outplanes/4), int(outplanes/4), kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(int(outplanes/4))
        self.conv3 = nn.Conv3d(int(outplanes/4), outplanes, kernel_size=1, bias=False)
        self.conv4 = nn.Conv3d(inplanes, outplanes , kernel_size=1, stride=stride, bias = False)
		
        self.bn3 = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.dropout = nn.Dropout3d(p=0.1, inplace=False)

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

        if (self.inplanes != self.outplanes) or (self.stride !=1 ):
            residual = self.conv4(x)

        out += residual
        out = self.relu(out)

        return out

class AttentionModule1(nn.Module):

    def __init__(self, in_channels, out_channels, size1=(64,32,32), size2=(32,16,16), size3=(16,8,8) , size4=(8,4,4)):
        super(AttentionModule1, self).__init__()
        self.first_residual_blocks = BottleneckAG(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = BottleneckAG(in_channels, out_channels)
        self.skip1_connection_residual_block = BottleneckAG(in_channels, out_channels)
        self.mpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = BottleneckAG(in_channels, out_channels)
        self.skip2_connection_residual_block = BottleneckAG(in_channels, out_channels)
        self.mpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = BottleneckAG(in_channels, out_channels)
        self.skip3_connection_residual_block = BottleneckAG(in_channels, out_channels)
        self.mpool4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax4_blocks = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
        )

        self.interpolation4 = nn.Upsample(size=size4,mode='trilinear')
        self.interpolation3 = nn.Upsample(size=size3,mode='trilinear')
        self.softmax5_blocks = BottleneckAG(in_channels, out_channels)
        self.interpolation2 = nn.Upsample(size=size2,mode='trilinear')
        self.softmax6_blocks = BottleneckAG(in_channels, out_channels)
        self.interpolation1 = nn.Upsample(size=size1,mode='trilinear')
        self.softmax7_blocks = BottleneckAG(in_channels, out_channels)
        self.softmax8_blocks = BottleneckAG(in_channels, out_channels)
        self.softmax9_blocks = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = BottleneckAG(in_channels, out_channels)

    def forward(self, x):
        mask = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(mask)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = out_softmax1
        out_softmax2 = self.skip1_connection_residual_block(out_softmax1)
        out_skip2_connection = out_softmax2

        out_mpool2 = self.mpool2(out_softmax2)
        out_softmax3 = self.softmax2_blocks(out_mpool2)
        out_skip3_connection = out_softmax3
        out_softmax4 = self.skip2_connection_residual_block(out_softmax3)
        out_skip4_connection = out_softmax4

        out_mpool3 = self.mpool3(out_softmax4)
        out_softmax5 = self.softmax3_blocks(out_mpool3)
        out_skip5_connection = out_softmax5
        out_softmax6 = self.skip3_connection_residual_block(out_softmax5)
        out_skip6_connection = out_softmax6		

        out_mpool4 = self.mpool4(out_softmax6)
        out_softmax7 = self.softmax4_blocks(out_mpool4)
        out_interp4 = self.interpolation4(out_softmax7) + out_skip6_connection
        out = out_interp4 + out_skip5_connection

        out_softmax8 = self.softmax5_blocks(out)
        out_interp3 = self.interpolation3(out_softmax8) + out_skip4_connection
        out = out_interp3 + out_skip3_connection
        out_softmax9 = self.softmax6_blocks(out)

        out_softmax10 = self.softmax7_blocks(out_softmax9)
        out_interp2 = self.interpolation2(out_softmax10) + out_skip2_connection
        out = out_interp2 + out_skip1_connection
        out_softmax11 = self.softmax8_blocks(out)

        out_interp1 = self.interpolation1(out_softmax11) + out_trunk
        out_softmax12 = self.softmax9_blocks(out_interp1)
        out = (1 + out_softmax12) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, size1=(32,16,16), size2=(16,8,8), size3=(8,4,4)):
        super(AttentionModule2, self).__init__()
        self.first_residual_blocks = BottleneckAG(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
         )
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = BottleneckAG(in_channels, out_channels)
        self.skip1_connection_residual_block = BottleneckAG(in_channels, out_channels)
        self.mpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = BottleneckAG(in_channels, out_channels)
        self.skip2_connection_residual_block = BottleneckAG(in_channels, out_channels)
        self.mpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
        )

        self.interpolation3 = nn.Upsample(size=size3,mode='trilinear')
        self.softmax4_blocks = BottleneckAG(in_channels, out_channels)
        self.interpolation2 = nn.Upsample(size=size2,mode='trilinear')
        self.softmax5_blocks = BottleneckAG(in_channels, out_channels)
        self.interpolation1 = nn.Upsample(size=size1,mode='trilinear')
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = BottleneckAG(in_channels, out_channels)

    def forward(self, x):
        mask = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(mask)

        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = out_softmax1
        out_softmax2 = self.skip1_connection_residual_block(out_softmax1)
        out_skip2_connection = out_softmax2
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax3 = self.softmax2_blocks(out_mpool2)
        out_skip3_connection = out_softmax3
        out_softmax4 = self.skip2_connection_residual_block(out_softmax3)
        out_skip4_connection = out_softmax4

        out_mpool3 = self.mpool3(out_softmax4)
        out_softmax5 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax5) + out_skip4_connection
        out = out_interp3 + out_skip3_connection

        out_softmax6 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax6) + out_skip2_connection
        out = out_interp2 + out_skip1_connection
        out_softmax7 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax7) + out_trunk
        out_softmax8 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax8) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule3(nn.Module):

    def __init__(self, in_channels, out_channels,  size1=(16,8,8), size2=(8,4,4)):
        super(AttentionModule3, self).__init__()
        self.first_residual_blocks = BottleneckAG(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = BottleneckAG(in_channels, out_channels)
        self.skip1_connection_residual_block = BottleneckAG(in_channels, out_channels)
        self.mpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = BottleneckAG(in_channels, out_channels)
        self.interpolation2 = nn.Upsample(size=size2,mode='trilinear')
        self.softmax3_blocks = BottleneckAG(in_channels, out_channels)
        self.interpolation1 = nn.Upsample(size=size1, mode='trilinear')
        self.softmax4_blocks = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = BottleneckAG(in_channels, out_channels)

    def forward(self, x):
        mask = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(mask)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = out_softmax1

        out_softmax2 = self.skip1_connection_residual_block(out_softmax1)
        out_skip2_connection = out_softmax2
        out_mpool2 = self.mpool2(out_softmax2)
        out_softmax3 = self.softmax2_blocks(out_mpool2)
        out_interp2 = self.interpolation2(out_softmax3) + out_skip2_connection
        out = out_interp2 + out_skip1_connection
        out_softmax4 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax4) + out_trunk
        out_softmax5 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax5) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule4(nn.Module):

    def __init__(self, in_channels, out_channels, size1=(8,4,4)):
        super(AttentionModule4, self).__init__()
        self.first_residual_blocks = BottleneckAG(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            BottleneckAG(in_channels, out_channels),
            BottleneckAG(in_channels, out_channels)
        )

        self.interpolation1 = nn.Upsample(size=size1,mode='trilinear')
        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = BottleneckAG(in_channels, out_channels)

    def forward(self, x):
        mask = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(mask)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last