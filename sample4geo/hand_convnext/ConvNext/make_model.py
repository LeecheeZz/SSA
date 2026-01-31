import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
import math


from sample4geo.Utils import init

class SE_Block(nn.Module):
    def __init__(self, inchannel=16, ratio=4):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  #  c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  #  c/r -> c
            nn.Sigmoid()
        )
        
    def forward(self, x):
            b, c, h, w = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            y = x * y.expand_as(x)
            return y
    
class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v

class SEAttention(nn.Module):
    def __init__(self, in_channels = 12):
        super(SEAttention, self).__init__()
        # self.h_se = SE_Block(inchannel = in_channels)
        # self.w_se = SE_Block(inchannel = in_channels)
        self.h_se = ECABlock(channels = in_channels)
        self.w_se = ECABlock(channels = in_channels)
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous() # H-SE
        x_out1 = self.h_se(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous() # W-SE
        x_out2 = self.w_se(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        return x_out11, x_out21
    
class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=5,padding=2,groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        ###### 调整顺序
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(11,1),padding=(5,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,11),padding=(0,5),groups=in_channels)
        
        # self.dconv1_3 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,15),padding=(0,7),groups=in_channels)
        # self.dconv3_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(15,1),padding=(7,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.ReLU()

    def forward(self, inputs):
        #   Global Perceptron
        # inputs = self.conv(inputs)
        # inputs = self.act(inputs)
        
        # channel_att_vec = self.ca(inputs)
        # inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        # x_3 = self.dconv1_3(x_init)
        # x_3 = self.dconv3_1(x_3)
        # x = x_1 + x_2 + x_3 + x_init
        x = x_1 + x_2 + x_init
        spatial_att = self.conv(x) # 1 * 1 卷积
        out = spatial_att * inputs
        out = self.conv(out)
        return out
    
class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale


# class TripletAttention(nn.Module):
#     def __init__(self):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()

#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         return x_out11, x_out21


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='kaiming'): # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block
        # self.tri_layer = TripletAttention()
        # self.se = SE_Block(inchannel=self.in_planes, ratio=4)
        # self.cbam = CBAM(in_channels=self.in_planes)
        # self.sk = SKAttention(channel = self.in_planes)
        # self.nl = _NonLocalBlockND(in_channels= self.in_planes)
        self.se_attention1 = ECABlock(channels=self.in_planes)
        self.se_attention2 = SEAttention(in_channels = 12)
        self.rep = RepBlock(in_channels=self.in_planes, out_channels=self.in_planes)
        # for i in range(self.block):
        #     name = 'classifier_mcb' + str(i + 1)
        #     setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        # -- backbone feature extractor
        gap_feature, part_features = self.convnext(x)

        main_feature = gap_feature.mean([-2, -1])
        # CFF
        tri_features = self.se_attention2(part_features) #[8, 768, 8, 8]  [8, 768, 8, 8]
        gap_feature = self.se_attention1(gap_feature) # [8, 768, 8, 8]
        trise_features = (gap_feature + tri_features[0] + tri_features[1]) / 3.0
        # # # DDC
        rep_features = self.rep(trise_features)
        # main_feature = rep_features.mean([-2, -1])
        # rep_features = self.sk(gap_feature)
        # -- Training
        if self.training:

            # 2. triplet attention
            # tri_features = self.tri_layer(part_features)  # 旋转另外两个轴，返回两个一样的特征体(bs, 1024, 12, 12); (bs, 1024, 12, 12)
            # convnext_feature = self.classifier1(trise_features.mean([-2, -1]))  # class: (bs, 701); feature: (bs, 512)  
            convnext_feature = self.classifier1(rep_features.mean([-2, -1]))  # class: (bs, 701); feature: (bs, 512)  
            # tri_list = []
            # for i in range(self.block):
            #     tri_list.append(tri_features[i].mean([-2, -1]))  # average pooling, 一张图变一个像素
            # triatten_features = torch.stack(tri_list, dim=2)  # 把另外两个轴旋转的特征体
            # if self.block == 0:
            #     y = []
            # else:
            #     y = self.part_classifier(self.block, triatten_features,
            #                              cls_name='classifier_mcb')  # 把另外两个轴旋转的feature也做分类
            # y = y + [convnext_feature]  # 三个分支连起来
            y = [convnext_feature]
            if self.return_f:  # return_f是triplet loss的设置，0.3
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return gap_feature, cls, features, main_feature, part_features

        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass

        # return gap_feature, part_features
        return main_feature, part_features

    # def part_classifier(self, block, x, cls_name='classifier_mcb'):
    #     part = {}
    #     predict = {}
    #     for i in range(block):
    #         part[i] = x[:, :, i].view(x.size(0), -1)
    #         name = cls_name + str(i + 1)
    #         c = getattr(self, name)
    #         predict[i] = c(part[i])
    #     y = []
    #     for i in range(block):
    #         y.append(predict[i])
    #     if not self.training:
    #         return torch.stack(y, dim=2)
    #     return y

    def fine_grained_transform(self):

        pass


def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class, block=block, return_f=return_f, resnet=resnet)
    return model
