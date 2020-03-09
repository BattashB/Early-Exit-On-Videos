import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .exit_blocks_reuse import EarlyExitBlockA,EarlyExitBlockB,EarlyExitBlockBS_LR_E1,EarlyExitBlockBS_LR_E2
import time
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



        
class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
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


class ResNextEE(nn.Module):

    def __init__(self,
                 opt,
                 block,
                 layers,
                 num_classes,
                 sample_size=112,
                 frames_sequence=16,
                 shortcut_type='B',
                 cardinality=32):
        self.inplanes = 64
        super(ResNextEE, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        inplanes_block1 = self.inplanes     
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        inplanes_block2 = self.inplanes         
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(frames_sequence / 16))
        last_size = int(math.ceil(sample_size / 32)) 
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        #print("inside ResnextEE, number of classes:",num_classes)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        self.th = opt.earlyexit_thresholds
        #self.exit1 = EarlyExitBlockA(block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2,num_classes)
        #self.exit2 = EarlyExitBlockB(block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2,num_classes)
        self.exit2_bool = opt.exit2
        self.exit1_bool = opt.exit1
        if opt.exit1:
            self.exit1 = EarlyExitBlockBS_LR_E1(block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2,num_classes)
            if opt.e1_weights:
                dict1 = torch.load(opt.e1_weights)['state_dict']
                new_dict={}
                for k,v in dict1.items():
                    if "exit1" in k:
                        new_dict[k[13:]]=v
                
                self.exit1.load_state_dict(new_dict)
                
        if opt.exit2:
            self.exit2 = EarlyExitBlockBS_LR_E2(block, layers, shortcut_type, cardinality,sample_size,frames_sequence,inplanes_block1,inplanes_block2,num_classes)            
            if opt.e2_weights:
                dict2 = torch.load(opt.e2_weights)['state_dict']
                new_dict2={}
                for k,v in dict2.items():
                    if "exit2" in k:
                        new_dict2[k[13:]]=v
                        
                self.exit2.load_state_dict(new_dict2)
        
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
        if stride != 1 or self.inplanes != planes * block.expansion:
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
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []

        exit_count = 0

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     
        
        
        x = self.layer1(x)##in:(1,64,8,28,28)
        if self.exit1_bool:
            exit1 = self.exit1(x)
            hold_exit1 = exit1
            exit1 = exit1[-1]
            max = torch.max(F.softmax(exit1,dim=0))
            argmax1 = torch.argmax(F.softmax(exit1,dim=0))
            exit_count = exit_count + 1
            if max.item() > self.th[exit_count - 1]:
                return exit1,exit_count-1
            


        #############################"""

        x = self.layer2(x)#in:(1,256,8,14,14)
        ##########exit2##############        
        #"""
        
        
        if self.exit2_bool:
            exit2 = self.exit2(x,hold_exit1[0],hold_exit1[1])
            exit2 = exit2[-1]
            max = torch.max(F.softmax(exit2,dim=0))
            argmax2 = torch.argmax(F.softmax(exit2,dim=0))
            exit_count = exit_count + 1
            
            if max.item() > self.th[exit_count - 1]:
                return exit2,exit_count - 1
            """
            #elif argmax1 == argmax2:# and  max.item() > (self.th[exit_count - 1]-0.1):
            elif argmax1 == argmax2 and  max.item() > (self.th[exit_count - 1]-0.01):
                print("Consistent")
                return exit2,exit_count - 1
           
            elif argmax1 == argmax2 and  max.item() > (self.th[exit_count - 1]-0.2):
                print("it's maybe not over the th but it's consistent")
                return exit2,exit_count - 1
            """

        #############################
        
        x = self.layer3(x)##in:(1,512,4,7,7)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
     
        return x,exit_count 


def get_fine_tuning_parameters(model, ft_begin_index):
    print("entered get_fine_tuning_parameters function")
    if ft_begin_index == 0:
        print("no need to further do on get_fine_tuning_parameters")
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        #print("those are k,v:",k,v.shape)
        for ft_module in ft_module_names:
            #print("now handeling this ft_module:",ft_module)
            if ft_module in k:
               # print("ft_module in k")
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(opt,num_classes,frame_size,frames_sequence,**kwargs):
    """Constructs a ResNet-101 model.
    """
   # print("inside calling func, number of classes:",num_classes)
    model = ResNextEE(opt,ResNeXtBottleneck, [3, 4, 23, 3],num_classes,frame_size,frames_sequence)
    return model



def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == "__main__":
    import torch
    #logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = ResNextEE(ResNeXtBottleneck, [3, 4, 23, 3])
    data = torch.autograd.Variable(torch.randn(1,3,16,112,112))
    output = net(data)
    torch.save({'state_dict': net.state_dict()}, './tmp.pth')
    print(output.shape)

