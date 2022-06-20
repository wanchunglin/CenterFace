# -*- coding: utf-8 -*-

from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
import torch.nn.functional as F


__all__ = ['MobileNetV2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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
            nn.ReLU6(inplace=False)
            #nn.ReLU()
        )

#1x1 expansion layer+batch normalization+relu6
#3x3 depthwise convolution+batch normalization+relu6
#1x1 projection layer+batch normalization
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

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

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

#conv2d
#bottleneck(block)*7
#conv2d 1x1
#avepool 7x7
#conv2d 1x1
class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.0,round_nearest=8,):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s     t=expansion ratio c=channel n=block repeated times s=stride
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
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for id,(t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id  :
                #setattr(object,name,value)
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

    def forward(self, x):
        y = []
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
                    nn.ReLU(inplace=True))
                    #nn.ReLU())
        # self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        # out = self.smooth(x + y)
        out = x + y
        return out
    
class LevelAttentionModel(nn.Module):
    def __init__(self,num_features_in,feature_size=256):
        super(LevelAttentionModel,self).__init__()
        self.conv1=nn.Conv2d(num_features_in,feature_size,kernal_size=3,padding=1)
        self.conv2=nn.Conv2d(feature_size,feature_size,kernal_size=3,padding=1)
        self.conv3=nn.Conv2d(feature_size,feature_size,kernal_size=3,padding=1)
        self.conv4=nn.Conv2d(feature_size,feature_size,kernal_size=3,padding=1)
        self.conv5=nn.Conv2d(feature_size,1,kernal_size=3,padding=1)
        
    def forward(self,x):
        out=self.conv1(x)
        out=nn.ReLU(out)
        out=self.conv2(out)
        out=nn.ReLU(out)
        out=self.conv3(out)
        out=nn.ReLU(out)
        out=self.conv4(out)
        out=nn.ReLU(out)
        out=self.conv5(out)
        out=nn.Sigmoid(out)
        
        return out
class LevelAttention_loss(nn.Module):
    def forward(self,img_batch_shape,attentions,bboxs):
        h,w=img_batch_shape[2],img_batch_shape[3]
        mask_losses=[]
        batch_size=bboxs.shape[0]
        for j in range(batch_size):
            bbox_attention=bboxs[j,:,:]
            bbox_attention=bbox_attention[bbox_attention[:,4]!=(-1)]
            
            cond1=torch.le(bbox_attention[:,0],w)
            cond2=torch.le(bbox_attention[:,1],h)
            cond3=torch.le(bbox_attention[:,2],w)
            cond4=torch.le(bbox_attention[:,3],h)
            cond=cond1*cond2*cond3*cond4
            bbox_attention=bbox_attention[cond,:]
            if bbox_attention.shape[0]==0:
                mask_losses.append(torch.tensor(0).float().cuda())
                continue
            bbox_area=(bbox_attention[:,2]-bbox_attention[:,0])*(bbox_attention[:,3]-bbox_attention[:,1])
            mask_loss=[]
            for id in range(len(attentions)):
                attention_map=attentions[id][j,0,:,:]
                min_area=(2**(id+5))**2*0.5
                max_area=(2**(id+5)*1.58)**2*2
                level_bbox_indice1=torch.ge(bbox_area,min_area)
                level_bbox_indice2=torch.le(bbox_area,max_area)
                level_bbox_indice=level_bbox_indice1*level_bbox_indice2
                level_bbox_annotation=bbox_attention[level_bbox_indice,:].clone()
                attention_h,attention_w=attentions.shape
                if level_bbox_annotation.shape[0]:
                    level_bbox_annotation[:,0]+=attention_w/w
                    level_bbox_annotation[:,1]+=attention_h/h
                    level_bbox_annotation[:,2]+=attention_w/w
                    level_bbox_annotation[:,3]+=attention_h/h
                mask_gt=torch.zero(attentions.shape[0])
                mask_gt=mask_gt.cuda()
                for i in range(level_bbox_annotation.shape[0]):
                    x1 = max(int(level_bbox_annotation[i,0]),0)
                    y1 = max(int(level_bbox_annotation[i,1]),0)
                    x2 = max(math.ceil(level_bbox_annotation[i,2])+1,attention_w)
                    y2 = max(math.ceil(level_bbox_annotation[i,3])+1,attention_h)
                    mask_gt[y1:y2,x1:x2]=1
                mask_gt=mask_gt[mask_gt>=0]
                mask_predict=attentions[attentions>=0]
                mask_loss.append(F.binary_cross_entropy(mask_predict,mask_gt))
            mask_loss.append(torch.stack(mask_loss).mean())
            
        return torch.stack(mask_losses).mean(dim=0,keepdim=True)
                
                    
                
            
        
        
        
class MobileNetUp(nn.Module):
    def __init__(self, channels, out_dim = 24):
        super(MobileNetUp, self).__init__()
        self.attentions=[]
        channels =  channels[::-1]
        self.conv =  nn.Sequential(
                    nn.Conv2d(channels[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))
        self.conv_last =  nn.Sequential(
                    nn.Conv2d(out_dim,out_dim,
                              kernel_size=3, stride=1, padding=1 ,bias=False),
                    nn.BatchNorm2d(out_dim,eps=1e-5,momentum=0.01),
                    nn.ReLU(inplace=True))
                    #nn.ReLU())

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
            self.attentions.append(x)
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
    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        ret = {}
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
                 pretrained=True,
                 head_conv=head_conv)
  return model


if __name__ == '__main__':
    import torch
    input = torch.zeros([1,3,416,416])
    model = get_mobile_net(10,{'hm':1, 'hm_offset':2, 'wh':2, 'landmarks':10},head_conv=24)          # hm reference for the classes of objects//这个头文件只能做矩形框检测
    res = model(input)
    print(res.shape)
