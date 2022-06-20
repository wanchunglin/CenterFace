from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
import torch


__all__ = ['MobileNetV2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}
from opts_pose import opts

opt=opts()
opt=opt.init()

attention=opt.attention
iterdet=opt.iterdet
concat=opt.concat
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            #nn.ReLU6(inplace=False)
            nn.ReLU()
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,atten=0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.atten=atten

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        if attention==1 or self.atten==1:
            self.ca = ChannelAttention(oup)
            self.sa = SpatialAttention()

    def forward(self, x):
        if self.use_res_connect:
            y=self.conv(x)
            if attention==1 or self.atten==1:
                y = self.ca(y) * y
                y = self.sa(y) * y
            #x+=y
            return x+y#x + self.conv(x)
        else:
            x=self.conv(x)
            if attention==1 or self.atten==1:
                x = self.ca(x) * x
                x = self.sa(x) * x
            return x#self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.0,round_nearest=8,):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 0
            [6, 24, 2, 2], # 1
            [6, 32, 3, 2], # 2
            [6, 64, 4, 2], # 3
            [6, 96, 3, 1], # 4
            [6, 160, 3, 2],# 5
            [6, 320, 1, 1],# 6
        ]
        self.feat_id = [1,2,4,6]
        self.feat_channel = []


        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        if iterdet:
            self.feature1=ConvBNReLU(3, input_channel, stride=2)
            self.history_transform=nn.Conv2d(1,input_channel,kernel_size=3,
                                             stride=2,padding=1,bias=True)
            features=[]
        else:
            features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for id,(t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                if attention==3:
                    
                    if i==len(range(n))-1:
                        atten=1
                        features.append(block(input_channel, output_channel, stride, expand_ratio=t,atten=atten))
                    else:
                        features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id  :
                self.__setattr__("feature_%d"%id,nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x,history_map=None):
        y = []
        if iterdet:
            history=self.history_transform(history_map)
            x=self.feature1(x)
            #print(x.size())
            #print(history.size())
            x=x+history
            for id in self.feat_id:
                x = self.__getattr__("feature_%d"%id)(x)
                y.append(x)
            
        else:    
            for id in self.feat_id:
                x = self.__getattr__("feature_%d"%id)(x)
                y.append(x)
        return y

def load_model(model,state_dict):
    new_model=model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)

def dict2list(func):
    def wrap(*args, **kwargs):
        self = args[0]
        x = args[1]
        ret_list = []
        ret = func(self, x)
        for k, v in ret[0].items():
            ret_list.append(v)
        return ret_list
    return wrap

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class IDAUp(nn.Module):
    def __init__(self, out_dim, channel):
        super(IDAUp, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                        output_padding=0, groups=out_dim, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU())
        self.conv =  nn.Sequential(
                    nn.Conv2d(channel, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    #nn.ReLU(inplace=False))
                    nn.ReLU())
        # self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        # out = self.smooth(x + y)
        out = x + y
        return out

class MobileNetUp(nn.Module):
    def __init__(self, channels, out_dim = 24):
        super(MobileNetUp, self).__init__()
        channels =  channels[::-1]
        self.conv =  nn.Sequential(
                    nn.Conv2d(channels[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=False))
        self.conv_last =  nn.Sequential(
                    nn.Conv2d(out_dim,out_dim,
                              kernel_size=3, stride=1, padding=1 ,bias=False),
                    nn.BatchNorm2d(out_dim,eps=1e-5,momentum=0.01),
                    #nn.ReLU(inplace=False)
                    nn.ReLU())

        for i,channel in enumerate(channels[1:]):
            setattr(self,'up_%d'%(i),IDAUp(out_dim,channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.ConvTranspose2d):
                fill_up_weights(m)

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        x = self.conv(layers[-1])

        for i in range(0,len(layers)-1):
            up = getattr(self, 'up_{}'.format(i))
            x = up([x,layers[len(layers)-2-i]])
        x = self.conv_last(x)
        return x

class MobileNetSeg(nn.Module):
    def __init__(self, base_name,heads,head_conv=24, pretrained = True):
        super(MobileNetSeg, self).__init__()
        self.heads = heads
        self.base = globals()[base_name](
            pretrained=pretrained)
        channels = self.base.feat_channel
        self.dla_up = MobileNetUp(channels, out_dim=head_conv)
        if attention==2:
            
            self.sa = SpatialAttention()
            if concat:
                self.concat_conv1=conv3x3(head_conv, head_conv)
                self.concat_conv2=conv3x3(head_conv, head_conv)
                self.concat_conv3=conv3x3(head_conv, head_conv)
                self.concat_conv4=conv3x3(head_conv, head_conv)
                self.ca = ChannelAttention(head_conv*3)
                
            else:
                self.ca = ChannelAttention(head_conv)
                
        if concat:
            for head in self.heads:
                classes = self.heads[head]
                if head == 'hm':
                    fc = nn.Sequential(
                        nn.Conv2d(head_conv*3, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True),
                        #nn.Sigmoid()
                    )
                else:
                    fc = nn.Conv2d(head_conv*3, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True)
                # if 'hm' in head:
                #     fc.bias.data.fill_(-2.19)
                # else:
                #     nn.init.normal_(fc.weight, std=0.001)
                #     nn.init.constant_(fc.bias, 0)
                self.__setattr__(head, fc)
        else:
            
            for head in self.heads:
                classes = self.heads[head]
                if head == 'hm':
                    fc = nn.Sequential(
                        nn.Conv2d(head_conv, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True),
                        #nn.Sigmoid()
                    )
                else:
                    fc = nn.Conv2d(head_conv, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True)
                # if 'hm' in head:
                #     fc.bias.data.fill_(-2.19)
                # else:
                #     nn.init.normal_(fc.weight, std=0.001)
                #     nn.init.constant_(fc.bias, 0)
                self.__setattr__(head, fc)

    # @dict2list         # 转onnx的时候需要将输出由dict转成list模式
    def forward(self, x,history_map=None):
        if iterdet:
            x = self.base(x,history_map)
        else:
            x = self.base(x)
        x = self.dla_up(x)
        ret = {}
        if attention==2:
            if concat:
                x=self.concat_conv1(x)
                x1=self.concat_conv2(x)
                x2=self.concat_conv3(x1)
                x2=self.concat_conv4(x2)
                x=torch.cat((x,x1,x2),dim=1)
            #print('Attention')
            x = self.ca(x) * x
            x = self.sa(x) * x
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def mobilenetv2_10(pretrained=True, **kwargs):
    model = MobileNetV2(width_mult=1.0)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                              progress=True)
        load_model(model,state_dict)
    return model

def mobilenetv2_5(pretrained=False, **kwargs):
    model = MobileNetV2(width_mult=0.5)
    if pretrained:
        print('This version does not have pretrain weights.')
    return model

# num_layers  : [10 , 5]
def get_mobile_net(num_layers, heads, head_conv=24):
  model = MobileNetSeg('mobilenetv2_{}'.format(num_layers), heads,
                 pretrained=False,
                 head_conv=head_conv)
  return model


if __name__ == '__main__':
    import torch
    input = torch.zeros([1,3,416,416])
    model = get_mobile_net(10,{'hm':1, 'hm_offset':2, 'wh':2, 'landmarks':10},head_conv=24)          # hm reference for the classes of objects//这个头文件只能做矩形框检测
    res = model(input)
    print(res.shape)
