import torch 
import torch.nn as nn 
import torch.nn.functional as F
import random
import numpy as np
from thop import profile
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, List
import numpy as np


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',}

class Dense_MLP(torch.nn.Module):
    def __init__(self,net_width,net_depth,tc,input_dims=2, num_classses=10,act_name='elu'):
        super(Dense_MLP, self).__init__()
        self.analyze_topology(net_width,net_depth,tc)
        batchnorm_list=[]
        layer_list=[]
        layer_list.append(nn.Linear(input_dims, net_width+self.layer_tc[0]))
        batchnorm_list.append(nn.BatchNorm1d(net_width+self.layer_tc[0]))
        self.net_depth=net_depth
        self.act=act_name
        if act_name =='relu':
            self.act_function = F.relu
        if act_name =='elu':
            self.act_function = F.elu


        for i in range(net_depth-2):
            layer_list.append(nn.Linear(net_width+self.layer_tc[i+1], net_width))
            batchnorm_list.append(nn.BatchNorm1d(net_width+self.layer_tc[i+1]))  
        layer_list.append(nn.Linear(net_width+self.layer_tc[net_depth-1], num_classses))
        self.features = nn.ModuleList(layer_list).eval() 
        self.batchnorm = nn.ModuleList(batchnorm_list).eval()
        self.link_dict=[]
        for i in range(net_depth):
            self.link_dict.append(self.add_link(i))
        
        input = torch.randn(1, input_dims)
        self.params,self.flops=self.param_num()
        self.macs, self.params = profile(self, inputs=(input, ))

    def param_num(self):
        num_param=0
        flops=0
        for layer in self.features:
            num_param=num_param+(layer.in_features)*(layer.out_features)+layer.out_features
            flops=flops+2*(layer.in_features)*(layer.out_features)+layer.out_features
        return num_param,flops
    def analyze_topology(self,net_width,net_depth,tc):
        all_path_num=np.zeros(net_depth)
        layer_tc=np.zeros(net_depth)
        for i in range(net_depth-2):
            for j in range(i+1):
                all_path_num[i+2]=all_path_num[i+2]+(net_width)
            layer_tc[i+2]=min(tc,all_path_num[i+2])

        self.layer_tc=np.array(layer_tc,dtype=int)
        self.all_path_num=np.array(all_path_num,dtype=int)
        self.density=(np.sum(layer_tc))/(np.sum(all_path_num))
        self.nn_mass=self.density*net_width*net_depth

    def add_link(self,idx=0):
        tmp=list((np.arange(self.all_path_num[idx])))
        link_idx=random.sample(tmp,self.layer_tc[idx])
        link_params=torch.tensor(link_idx)
        return link_params

    def forward(self, x):
        out0=self.act_function(self.features[0](x))        
        out1=self.features[1](out0)
        out_dict=[]
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict=[]
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1,out0),1))
        for layer_idx in range(self.net_depth-2):
            in_features=feat_dict[layer_idx]
            if self.layer_tc[layer_idx+2]>0:
                in_tmp=torch.cat((out_dict[layer_idx+1],in_features[:,self.link_dict[layer_idx+2]]),1)
                if layer_idx<self.net_depth-3:
                    out_tmp=self.features[layer_idx+2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)           
            else:
                in_tmp=out_dict[layer_idx+1]
                if layer_idx<self.net_depth-3:
                    out_tmp=self.features[layer_idx+2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)     
        return out_dict[self.net_depth-1]


    def dev(self,fc_layer,layer_idx,x):
        batch_num=x.size()[0]
        if self.act=='relu' and layer_idx>1:
            w=fc_layer.weight.data
            dims=w.size()
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)
            one = torch.ones_like(x)
            relu_dev=torch.where(x > 0, one, zero)
            b=torch.zeros([batch_num,dims[1],dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i,j,j]=relu_dev[i,j]
            j=torch.matmul(W,b)
            _,sigma,_=torch.svd(j)
            return sigma
        
        if self.act=='elu' and layer_idx>1:
            w=fc_layer.weight.data
            dims=w.size()
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)
            one = torch.exp(x)
            relu_dev=torch.where(x > 0, one, zero)
            b=torch.zeros([batch_num,dims[1],dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i,j,j]=relu_dev[i,j]
            j=torch.matmul(W,b)
            _,sigma,_=torch.svd(j)
            return sigma

        if layer_idx==1:
            w=fc_layer.weight.data
            dims=w.size()

            W=w.unsqueeze(0).repeat(batch_num,1,1)
            _,sigma,_=torch.svd(W)
            return sigma
        if layer_idx==0:
            w=fc_layer.weight.data
            dims=w.size()

            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)

            if self.act=='relu':            
                one = torch.ones_like(x)
                relu_dev=torch.where(x > 0, one, zero)
            if self.act=='elu':            
                one = torch.exp(x)
                relu_dev=torch.where(x > 0, one, zero)

            b=torch.zeros([batch_num,dims[0],dims[0]])
            for i in range(batch_num):
                for j in range(dims[0]):
                    b[i,j,j]=relu_dev[i,j]


            j=torch.matmul(b,W)
            _,sigma,_=torch.svd(j)
            return sigma

    def isometry(self, x):
        out0=self.act_function(self.features[0](x))        
        out1=self.features[1](out0) 

        in_dict=[]
        in_dict.append(self.features[0](x))
        in_dict.append(out0)

        out_dict=[]
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict=[]
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1,out0),1))
        for layer_idx in range(self.net_depth-2):
            in_features=feat_dict[layer_idx]
            if self.layer_tc[layer_idx+2]>0:
                in_tmp=torch.cat((out_dict[layer_idx+1],in_features[:,self.link_dict[layer_idx+2]]),1)
                in_dict.append(in_tmp)
                if layer_idx<self.net_depth-3:
                    out_tmp=self.features[layer_idx+2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)           
            else:
                in_tmp=out_dict[layer_idx+1]
                in_dict.append(in_tmp)
                if layer_idx<self.net_depth-3:
                    out_tmp=self.features[layer_idx+2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)  
        sigma_all=None
        for i in range(self.net_depth):
            sigma=self.dev(self.features[i],layer_idx=i,x=in_dict[i])   
            if i==0:
                sigma_all=sigma
            else:
                sigma_all=torch.cat((sigma_all,sigma),1)
        
        sig_mean=sigma_all.view(-1).mean()
        sig_std=sigma_all.view(-1).std()
        return sig_mean,sig_std



def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
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
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        padding = (kernel_size - 1) // 2
        self.nn_mass=0
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )



class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if self.use_res_connect:
            self.nn_mass=  (inp+2*hidden_dim)  *inp/(2*inp+hidden_dim)

        else:
            self.nn_mass=0
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.expand_ratio=expand_ratio

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def get_nn_mass(self):
        self.nn_mass = 0
        for mode in self.features:
            self.nn_mass = self.nn_mass + mode.nn_mass
            # print(mode.nn_mass)
        # print(self.features)
        return self.nn_mass

  
    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, wm, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.expansion = 1

        self.nn_mass = (planes+inplanes)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = torch.zeros_like(x)
            identity[:,self.shortcut_idx,:,:] = x[:,self.shortcut_idx,:,:]

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, wm, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.nn_mass=(inplanes + 2*width)  *inplanes/(2*inplanes+width)
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = torch.zeros_like(x)
            identity[:,self.shortcut_idx,:,:] = x[:,self.shortcut_idx,:,:]


        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, wm, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(wm, block, 64, layers[0])
        self.layer2 = self._make_layer(wm, block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(wm, block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(wm, block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        nn_mass=0
        for block in self.layer1:
            nn_mass=nn_mass+block.nn_mass
        for block in self.layer2:
            nn_mass=nn_mass+block.nn_mass
        for block in self.layer3:
            nn_mass=nn_mass+block.nn_mass
        for block in self.layer4:
            nn_mass=nn_mass+block.nn_mass
        self.nn_mass=nn_mass

    def _make_layer(self, wm, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(wm, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(wm, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def count_params(self):
        num_prams = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return ','#num_prams

def _resnet(arch, wm, block, layers, pretrained, progress, **kwargs):
    model = ResNet(wm, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', wm, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', wm, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', wm, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet101(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', wm, Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)



def resnet152(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', wm, Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)



def resnext50_32x4d(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', wm, Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def resnext101_32x8d(wm=1.0, pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', wm, Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)



def wide_resnet50_2(wm=1.0, pretrained=False, progress=True, **kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', wm, Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def wide_resnet101_2(wm=1.0, pretrained=False, progress=True, **kwargs):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', wm, Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class conv_depth(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=32,kernel_size=[3,1], stride=[1,1], padding=[1,0],bias=[False,False]):
        super(conv_depth, self).__init__()
        self.conv0=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
                            stride=1, padding=0,groups=1,bias=False)
        self.batch0=nn.BatchNorm2d(out_channels)
        self.conv1=nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size[0], 
                            stride=stride[0], padding=padding[0],groups=int(out_channels),bias=bias[0])

        self.batch1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size[1], 
                    stride=stride[1], padding=padding[1],groups=1,bias=bias[1])        
        self.batch2=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out=F.relu(self.batch0(self.conv0(x)))
        out=F.relu(self.batch1(self.conv1(out)))
        out=self.batch2(self.conv2(out))
        return out

class depth_wise_dense_cnn(torch.nn.Module):
    def __init__(self, cell_depth, wm, tc_array, num_cells=3, num_classes=10):
        super(depth_wise_dense_cnn, self).__init__( )
        cell_width=16*wm*np.power(2, np.arange(num_cells))
        self.width_base = cell_width
        self.link_dict,self.layer_tc = self.add_link(num_cells, cell_depth, cell_width, tc_array)
        self.cell_depth =cell_depth
        self.num_cells = num_cells

        self.layer_list = []
        self.norm_list = []
        input_channels = 3
        for i in range(num_cells):
            for j in range(cell_depth):
                if i ==0 and j == 0 :
                    self.layer_list.append(nn.Conv2d(in_channels=input_channels + self.layer_tc[i][j], out_channels=self.width_base[i], 
                                    kernel_size=3, stride=1, padding=1,bias=False) )
                elif j == 0:
                    self.layer_list.append(conv_depth(in_channels=self.width_base[i-1], out_channels=self.width_base[i],stride=[2,1]))
                else:
                    self.layer_list.append(conv_depth(in_channels=self.width_base[i]+self.layer_tc[i][j], out_channels=self.width_base[i]))                    
                self.norm_list.append( nn.BatchNorm2d(self.width_base[i]+self.layer_tc[i][j]) )
                input_channels = cell_width[i]
        self.layer_list=nn.ModuleList(self.layer_list).eval() 
        self.norm_list = nn.ModuleList(self.norm_list).eval() 

        self.avg_pool=nn.AvgPool2d(kernel_size=2,  padding=0)
        self.fc_layer=nn.Linear(in_features=self.width_base[2]*4*4, out_features=num_classes, bias=True)

    def forward(self, input_tensor):
        out_dict = input_tensor
        for cell_idx in range(self.num_cells):
            base_layer_idx = cell_idx*self.cell_depth
            out0=self.layer_list[0 +base_layer_idx](out_dict)
            out0=self.norm_list[0 +base_layer_idx](out0)

            out1=F.relu(out0)    
            out1=self.layer_list[1 + base_layer_idx](out1)
            out1=self.norm_list[1 +base_layer_idx](out1)

            out_dict=out1

            feat_dict=torch.cat((out1,out0),1)

            for layer_idx in range(base_layer_idx, base_layer_idx+ self.cell_depth-2):

                if self.layer_tc[cell_idx, layer_idx+2-base_layer_idx]>0:
                    in_tmp=torch.cat((out_dict,feat_dict[:,self.link_dict[cell_idx][layer_idx+2 - base_layer_idx],:,:]),1)

                    out_tmp = self.norm_list[layer_idx+2](in_tmp)
                    out_tmp=self.layer_list[layer_idx+2](F.relu(out_tmp))
                    feat_dict=torch.cat((out_tmp,feat_dict),1)
                    out_dict=out_tmp
     
                else:
                    out_tmp = self.norm_list[layer_idx+2](out_dict)
                    out_tmp=self.layer_list[layer_idx+2](F.relu(out_dict))
                    feat_dict=torch.cat((out_tmp,feat_dict),1)
                    out_dict=out_tmp

        out= torch.flatten(self.avg_pool(out_dict), 1)
        out=self.fc_layer(out)
        return out

    def add_link(self, num_cells, cell_depth, cell_width,tc_array):
        random.seed(2)
        all_path_num=np.zeros([num_cells,cell_depth])
        layer_tc=np.zeros([num_cells,cell_depth])
        nn_mass=0
        density=np.zeros(num_cells)
        cell_depth=cell_depth
        for k in range(num_cells):
            for i in range(cell_depth-2):
                for j in range(i+1):
                    all_path_num[k,i+2]=all_path_num[k,i+2]+cell_width[k]
                layer_tc[k,i+2]=min(tc_array[k],all_path_num[k,i+2])
            layer_tc=np.array(layer_tc,dtype=int)
            all_path_num=np.array(all_path_num,dtype=int)
            density[k]=(np.sum(layer_tc[k]))/(np.sum(all_path_num[k]))
            nn_mass=nn_mass+density[k]*cell_width[k]*cell_depth
        self.nn_mass = nn_mass
        self.density = density

        layer_tc=np.array(layer_tc,dtype=int)
        all_path_num=np.array(all_path_num,dtype=int)

        link_dict = []
        for cell_idx in range(num_cells):
            tmp_link_dict = []
            for idx in range(cell_depth):
                tmp=list((np.arange(all_path_num[cell_idx,idx])))

                link_idx=random.sample(tmp,layer_tc[cell_idx,idx])
                link_params=torch.tensor(link_idx,dtype=int)
                tmp_link_dict.append(link_params)
            link_dict.append(tmp_link_dict)
        return link_dict, layer_tc


class regular_dense_cnn(torch.nn.Module):
    def __init__(self, cell_depth, wm, tc_array, num_cells=3, num_classes=10):
        super(regular_dense_cnn, self).__init__( )
        cell_width=16*wm*np.power(2, np.arange(num_cells))
        self.width_base = cell_width
        self.link_dict,self.layer_tc = self.add_link(num_cells, cell_depth, cell_width, tc_array)
        self.cell_depth =cell_depth
        self.num_cells = num_cells

        self.layer_list = []
        self.norm_list = []
        input_channels = 3
        for i in range(num_cells):
            for j in range(cell_depth):
                if i ==0 and j==0:
                    self.layer_list.append(nn.Conv2d(in_channels=input_channels + self.layer_tc[i][j], out_channels=self.width_base[i], 
                                    kernel_size=3, stride=1, padding=1,bias=False) )
                elif j == 0 :
                    self.layer_list.append(nn.Conv2d(in_channels=input_channels + self.layer_tc[i][j], out_channels=self.width_base[i], 
                                    kernel_size=3, stride=2, padding=1,bias=False) )
                else:
                    self.layer_list.append(nn.Conv2d(in_channels=input_channels + self.layer_tc[i][j], out_channels=self.width_base[i], 
                                    kernel_size=3, stride=1, padding=1,bias=False) )                    
                self.norm_list.append( nn.BatchNorm2d(self.width_base[i]+self.layer_tc[i][j]) )
                input_channels = cell_width[i]
        self.layer_list=nn.ModuleList(self.layer_list).eval() 
        self.norm_list = nn.ModuleList(self.norm_list).eval() 
        self.avg_pool=nn.AvgPool2d(kernel_size=2,  padding=0)
        self.fc_layer=nn.Linear(in_features=self.width_base[2]*4*4, out_features=num_classes, bias=True)

    def forward(self, input_tensor):
        out_dict = input_tensor
        for cell_idx in range(self.num_cells):
            base_layer_idx = cell_idx*self.cell_depth
            out0=self.layer_list[0 +base_layer_idx](out_dict)
            out0=self.norm_list[0 +base_layer_idx](out0)

            out1=F.relu(out0)    
            out1=self.layer_list[1 + base_layer_idx](out1)
            out1=self.norm_list[1 +base_layer_idx](out1)

            out_dict=out1

            feat_dict=torch.cat((out1,out0),1)

            for layer_idx in range(base_layer_idx, base_layer_idx+ self.cell_depth-2):

                if self.layer_tc[cell_idx, layer_idx+2-base_layer_idx]>0:
                    in_tmp=torch.cat((out_dict,feat_dict[:,self.link_dict[cell_idx][layer_idx+2 - base_layer_idx],:,:]),1)
                    out_tmp = self.norm_list[layer_idx+2](in_tmp)
                    out_tmp=self.layer_list[layer_idx+2](F.relu(out_tmp))
                    feat_dict=torch.cat((out_tmp,feat_dict),1)
                    out_dict=out_tmp
     
                else:
                    out_tmp = self.norm_list[layer_idx+2](out_dict)
                    out_tmp=self.layer_list[layer_idx+2](F.relu(out_dict))
                    feat_dict=torch.cat((out_tmp,feat_dict),1)
                    out_dict=out_tmp


        out=self.avg_pool(out_dict)
        out =torch.flatten(out,1)
        out=self.fc_layer(out)
        return out

    def add_link(self, num_cells, cell_depth, cell_width,tc_array):
        random.seed(2)

        all_path_num=np.zeros([num_cells,cell_depth])
        layer_tc=np.zeros([num_cells,cell_depth])
        nn_mass=0
        density=np.zeros(num_cells)
        self.density = density
        cell_depth=cell_depth
        for k in range(num_cells):
            for i in range(cell_depth-2):
                for j in range(i+1):
                    all_path_num[k,i+2]=all_path_num[k,i+2]+cell_width[k]
                layer_tc[k,i+2]=min(tc_array[k],all_path_num[k,i+2])
            layer_tc=np.array(layer_tc,dtype=int)
            all_path_num=np.array(all_path_num,dtype=int)
            density[k]=(np.sum(layer_tc[k]))/(np.sum(all_path_num[k]))
            nn_mass=nn_mass+density[k]*cell_width[k]*cell_depth
        self.nn_mass = nn_mass
        layer_tc=np.array(layer_tc,dtype=int)
        all_path_num=np.array(all_path_num,dtype=int)

        link_dict = []
        for cell_idx in range(num_cells):
            tmp_link_dict = []
            for idx in range(cell_depth):
                tmp=list((np.arange(all_path_num[cell_idx,idx])))

                link_idx=random.sample(tmp,layer_tc[cell_idx,idx])
                link_params=torch.tensor(link_idx,dtype=int)
                tmp_link_dict.append(link_params)
            link_dict.append(tmp_link_dict)
        return link_dict, layer_tc





