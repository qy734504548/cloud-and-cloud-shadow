import torch
import torch.nn as nn
import torch.nn.functional as F
from Segmentation.model.SpCSAMNet_new.resnet import resnet50
from Segmentation.model.SpCSAMNet_new.cbam import ChannelGate,SpatialGate


# def double_con3x3(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )

def in_conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# class RRB(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(RRB, self).__init__()
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, input):
#         input = self.conv1x1(input)
#         x = self.conv3x3(input)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv3x3(x)
#         output = x+input
#         output = self.relu(output)
#         return output

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super(GCN, self).__init__()
        pad0 = int((kernel_size[0] - 1) // 2)
        pad1 = int((kernel_size[1] - 1) // 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = torch.cat((x_l,x_r),dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


def conv1x1_bn_rl(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv1x1_rl(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),

        nn.ReLU(inplace=True)
    )


class CSAM(nn.Module):
    def __init__(self, in_channels):
        super(CSAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels=in_channels, reduction_ratio=16, pool_types=['avg', 'max'])
        self.SpatialGate = SpatialGate()

    def forward(self, x1,x2):  # x1=high , x2 = low
        x = torch.cat([x1,x2],dim=1)
        x = self.ChannelGate(x)
        x2 = x * x2
        x3 = self.SpatialGate(x2)
        x3 = x3 * x2
        res = x3 + x1
        return res

class CSAMNet(nn.Module):
    def __init__(self, num_class):
        super(CSAMNet, self).__init__()
        resnet = resnet50(pretrained=False,spm_on=True)
        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer0 = in_conv3x3(in_channels=3, out_channels=64)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.CSAM1 = CSAM(in_channels=1024)
        self.CSAM2 = CSAM(in_channels=512)
        self.CSAM3 = CSAM(in_channels=256)
        self.CSAM4 = CSAM(in_channels=128)
        self.CSAM5 = CSAM(in_channels=64)

        self.conv1x1_1 = conv1x1_rl(in_channels=2048, out_channels=1024)
        self.GCN_1 = GCN(in_channels=2048, out_channels=512,kernel_size=(7,7))
        self.GCN_2 = GCN(in_channels=1024, out_channels=256, kernel_size=(7,7))
        self.GCN_3 = GCN(in_channels=512, out_channels=128, kernel_size=(7,7))
        self.GCN_4 = GCN(in_channels=256, out_channels=64, kernel_size=(7,7))
        self.GCN_5 = GCN(in_channels=64, out_channels=32, kernel_size=(7,7))

        self.conv1x1_a = conv1x1_bn_rl(in_channels=1024, out_channels=512)
        self.conv1x1_b = conv1x1_bn_rl(in_channels=512, out_channels=256)
        self.conv1x1_c = conv1x1_bn_rl(in_channels=256, out_channels=128)
        self.conv1x1_d = conv1x1_bn_rl(in_channels=128, out_channels=64)
        self.conv1x1_e = conv1x1_bn_rl(in_channels=64, out_channels=num_class)


    def forward(self, input):
        x1 = self.layer0(input)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x1 = self.GCN_5(x1)
        x2 = self.GCN_4(x2)
        x3 = self.GCN_3(x3)
        x4 = self.GCN_2(x4)
        x5 = self.GCN_1(x5)
        x6 = self.conv1x1_1(x6)
        x6 = F.interpolate(x6, size=x5.size()[2:], mode='bilinear', align_corners=True)
        x = self.CSAM1(x6,x5)
        x = self.conv1x1_a(x)
        x = F.interpolate(x, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = self.CSAM2(x, x4)
        x = self.conv1x1_b(x)
        x = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = self.CSAM3(x, x3)
        x = self.conv1x1_c(x)
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = self.CSAM4(x, x2)
        x = self.conv1x1_d(x)
        x = self.CSAM5(x, x1)
        x = self.conv1x1_e(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

if __name__ =='__main__':
    x = torch.randn(1,3,224,224)
    model = CSAMNet(3)
    print(model)
    y = model(x)
    print(y.shape)