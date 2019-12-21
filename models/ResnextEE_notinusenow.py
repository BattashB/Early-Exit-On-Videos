import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.resnext import ResNeXt,ResNeXtBottleneck

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

class MF_UNIT(nn.Module):

    def __init__(self, in_channels, num_mid, out_channels, g=1, stride=(1,1,1)):
        super(MF_UNIT, self).__init__()
        
        num_ix = int(num_mid/4)
        # prepare input
        self.conv_i1 =     Conv3dEE(in_channels=in_channels,  out_channels=num_ix,  kernel_size=(1,1,1), padding=(0,0,0))
        self.conv_i2 =     Conv3dEE(in_channels=num_ix,  out_channels=in_channels,  kernel_size=(1,1,1), padding=(0,0,0))
        self.conv_m1 =     Conv3dEE(in_channels=in_channels,  out_channels=num_mid, kernel_size=(3,3,3), padding=(1,1,1), stride=stride, g=g)
        self.conv_m2 =     Conv3dEE(in_channels=num_mid, out_channels=out_channels, kernel_size=(1,3,3), padding=(0,1,1), g=g)
        self.conv_w1 = Conv3dEE(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1,1,1), padding=(0,0,0), stride=stride)


    def forward(self, x):

        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        x = self.conv_w1(x)

        return h + x

        
class EarlyExitBlockA(nn.Module):
    def __init__(self, num_classes=51):
        super(EarlyExitBlockA, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        no_stride =(1,1,1)

        stride = (2,2,2)
        self.conv1_exit0 = MF_UNIT(64, 128,64,  stride=stride, g=16)#(in_c,out_c)
        self.conv2_exit0 = MF_UNIT(64, 128,64,  stride=no_stride, g=16)
        self.conv3_exit0 = MF_UNIT(64, 128,64,  stride=no_stride, g=16)
        self.conv4_exit0 = MF_UNIT(64, 128,64,  stride=stride, g=16)
        
        self.fc_exit0 = nn.Linear(64, num_classes)
        self.local_avgpool = nn.AvgPool3d((2, 7, 7), stride=1)
        
    def forward(self, x):
        exit0 = self.conv1_exit0(x)
        exit0 = self.conv2_exit0(exit0)
        exit0 = self.conv3_exit0(exit0)
        exit0 = self.conv4_exit0(exit0)
        exit0 = self.local_avgpool(exit0)
        exit0 = torch.squeeze(exit0)
        exit0 = self.fc_exit0(exit0)
        return exit0


class EarlyExitBlockB(nn.Module):
    def __init__(self, num_classes=51):
        super(EarlyExitBlockB, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        no_stride =(1,1,1)
        stride = (2,2,2)
        self.conv1_exit0 = MF_UNIT(256, 128,256,  stride=stride, g=16)
        self.conv2_exit0 = MF_UNIT(256, 128,256,  stride=no_stride, g=16)
        self.conv3_exit0 = MF_UNIT(256, 128,256,  stride=no_stride, g=16)
        self.conv4_exit0 = MF_UNIT(256, 128,64,  stride=stride, g=16)
        
        self.fc_exit0 = nn.Linear(64, num_classes)
        self.local_avgpool = nn.AvgPool3d((2, 7, 7), stride=1)
        
    def forward(self, x):
        exit0 = self.conv1_exit0(x)
        exit0 = self.conv2_exit0(exit0)
        exit0 = self.conv3_exit0(exit0)
        exit0 = self.conv4_exit0(exit0)
        exit0 = self.local_avgpool(exit0)
        exit0 = torch.squeeze(exit0)
        exit0 = self.fc_exit0(exit0)
        return exit0

class EarlyExitBlockC(nn.Module):
    def __init__(self, num_classes=51):
        super(EarlyExitBlockC, self).__init__()
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
        no_stride =(1,1,1)
        self.conv1_exit0 = MF_UNIT(512, 1024,512,  stride=no_stride, g=16)
        self.conv2_exit0 = MF_UNIT(512, 1024,512,  stride=no_stride, g=16)
        self.conv3_exit0 = MF_UNIT(512, 1024,512,  stride=no_stride, g=16)
        self.conv4_exit0 = MF_UNIT(512, 1024,512,  stride=no_stride, g=16)
        self.conv5_exit0 = MF_UNIT(512, 1024,512,  stride=no_stride, g=16)
        self.conv6_exit0 = MF_UNIT(512, 1024,512,  stride=no_stride, g=16)
        self.conv7_exit0 = MF_UNIT(512, 128,128,  stride=stride, g=16)
        
        self.fc_exit0 = nn.Linear(128, num_classes)
        self.local_avgpool = nn.AvgPool3d((2, 7, 7), stride=1)
        
    def forward(self, x):
        exit0 = self.conv1_exit0(x)
        exit0 = self.conv2_exit0(exit0)
        exit0 = self.conv3_exit0(exit0)
        exit0 = self.conv4_exit0(exit0)
        exit0 = self.conv5_exit0(exit0)
        exit0 = self.conv6_exit0(exit0)
        exit0 = self.conv7_exit0(exit0)
        exit0 = self.local_avgpool(exit0)
        exit0 = torch.squeeze(exit0)
        exit0 = self.fc_exit0(exit0)
        return exit0


class ResNextEE(ResNeXt):
    def __init__(self, block, layers,frame_size,frames_sequence, num_classes=51):
        super(ResNextEE, self).__init__(block, layers,frame_size,frames_sequence, num_classes)

        # Define early exit layers
        ksize=(3,3,3)
        pad=(1,1,1)
        nt_stride =(1,2,2)
        stride = (2,2,2)
       
        self.exit0 = EarlyExitBlockA(num_classes)
        self.exit1 = EarlyExitBlockB(num_classes)
        self.exit2 = EarlyExitBlockC(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #exit0 = self.exit0(x)
        x = self.layer1(x)##in:(1,64,8,28,28)
        #exit1 = self.exit1(x)
        x = self.layer2(x)##in:(1,256,8,28,28)
        exit = self.exit2(x)
        x = self.layer3(x)##in:(1,512,4,14,14)
        x = self.layer4(x)##in:(1,1024,2,7,7)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        #return x

        output = []
        output.append(exit)
        #output.append(exit1)
        output.append(x)
        return output


def resnext50(frame_size,frames_sequence,**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNextEE(ResNeXtBottleneck, [3, 4, 6, 3],frame_size,frames_sequence, **kwargs)
    return model


def resnext101(frame_size,frames_sequence,**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNextEE(ResNeXtBottleneck, [3, 4, 23, 3],frame_size,frames_sequence, **kwargs)
    return model


def resnext152(frame_size,frames_sequence,**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNextEE(ResNeXtBottleneck, [3, 8, 36, 3],frame_size,frames_sequence, **kwargs)
    return model

if __name__ == "__main__":
    import torch
    #logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = ResNextEE(ResNeXtBottleneck, [3, 4, 23, 3])
    data = torch.autograd.Variable(torch.randn(1,3,16,112,112))
    output = net(data)
    torch.save({'state_dict': net.state_dict()}, './tmp.1h')
    print(output.shape)


