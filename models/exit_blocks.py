import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
        

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class Conv3dEE(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1), g=1, bias=False):
        super(Conv3dEE, self).__init__()
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                               stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(self.bn(h))
        return h



class EarlyExitBlockA(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockA, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = 256
        self.block1 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.inplanes = 512
        self.block2 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = 1024
        self.block3 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)    
        
    def forward(self, x):
        exit1 = self.block1(x)
        exit1 = self.block2(exit1)
        exit1 = self.block3(exit1)
        exit1 = self.avgpool(exit1)        
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)

        return exit1




class EarlyExitBlockB(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockB, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = inplanes_block1
        self.block1 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = inplanes_block2
        self.block2 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)    
        
    def forward(self, x):
        exit1 = self.block1(x)
        exit1 = self.block2(exit1)
        exit1 = self.avgpool(exit1)
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)
        
        return exit1

class ResNeXtBottleneckEE(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneckEE, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        #self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class EarlyExitBlockBS_LR_E1(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockBS_LR_E1, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = int(inplanes_block1/2)
        self.block0 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.block1 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = inplanes_block2
        self.block2 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        block = ResNeXtBottleneckEE
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        groups=16,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)    
    def forward(self, x):
        out = []
        exit1 = self.block0(x)
        exit1 = self.block1(exit1)
        out.append(exit1)
        exit1 = self.block2(exit1)
        out.append(exit1)
        exit1 = self.avgpool(exit1)
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)
        out.append(exit1)
        return out
        
class EarlyExitBlockBS_LR_E2(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockBS_LR_E2, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = inplanes_block1

        self.block1 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = inplanes_block2
        self.block2 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        block = ResNeXtBottleneckEE

        
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        groups=16,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)    

    
    def forward(self, x):
        out = []
        exit1 = self.block1(x)
        out.append(exit1)
        exit1 = self.block2(exit1)
        out.append(exit1)
        exit1 = self.avgpool(exit1)
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)
        out.append(exit1)
        
        return out
                
class EarlyExitBlockBS(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockBS, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = inplanes_block1

        self.block1 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = inplanes_block2
        self.block2 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)    
    
    def forward(self, x):
        out = []
        exit1 = self.block1(x)
        out.append(exit1)
        exit1 = self.block2(exit1)
        out.append(exit1)
        exit1 = self.avgpool(exit1)
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)
        out.append(exit1)
        
        return out
        
class EarlyExitBlockLR_E1(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockLR_E1, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = int(inplanes_block1/2)
        self.block0 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.block1 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = inplanes_block2
        self.block2 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        block = ResNeXtBottleneckEE
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        groups=16,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)    
    def forward(self, x):
        out = []
        exit1 = self.block0(x)
        exit1 = self.block1(exit1)
        exit1 = self.block2(exit1)
        exit1 = self.avgpool(exit1)
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)
        out.append(exit1)
        return out
        
class EarlyExitBlockLR_E2(nn.Module):
    def __init__(self,block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2, num_classes=51):
        super(EarlyExitBlockLR_E2, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        self.inplanes = inplanes_block1

        self.block1 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.inplanes = inplanes_block2
        self.block2 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        block = ResNeXtBottleneckEE

        
        if shortcut_type == 'A':
            downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
        else:
            downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        groups=16,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)    

    
    def forward(self, x):
        out = []
        exit1 = self.block1(x)
        exit1 = self.block2(exit1)
        exit1 = self.avgpool(exit1)
        exit1 = torch.squeeze(exit1)
        exit1 = self.dropout(exit1)
        exit1 = self.fc(exit1)
        out.append(exit1)
        
        return out
