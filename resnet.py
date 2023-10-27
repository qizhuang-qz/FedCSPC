import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


    
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super(DNN_v5, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], n_classes)
        

    def forward(self, x):
        h1 = self.l1(x)
        h1 = F.relu(h1)
        h2 = self.l2(h1)
        out = self.l3(h2)
        return h2, out       
        
    
    

__all__ = ['resnet18', 'resnet34', 'resnet50']



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}




def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv3x3_bn(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    modules = nn.Sequential(
        nn.BatchNorm2d(in_planes),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
    )
    return modules


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1_bn(in_planes, out_planes, stride=1, groups=1):
    modules = nn.Sequential(
        nn.BatchNorm2d(in_planes),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False),
    )
    return modules


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, kernel_size, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=kernel_size, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.l1= nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.l2= nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.l3 = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.groups, self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, PH=True):
        x = self.conv1(x)
        x = self.bn1(x)
#         x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = self.avgpool(F.relu(out4))
        feats = out.view(out.size(0), -1)
        feats_out = self.fc(feats)
        feats_lin1 = self.l1(feats)
        feats_lin2 = self.l2(feats_lin1)
        out = self.l3(feats_lin2)
        return feats, feats_out, feats_lin2, out

 
    

def resnet18(dataset, kernel_size=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = ResNet(BasicBlock, kernel_size, [2, 2, 2, 2])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        print('done load model')
    else:
        model = ResNet(BasicBlock, kernel_size, [2, 2, 2, 2], **kwargs)
    if dataset == 'tinyimagenet':
        model.l3 = nn.Linear(model.l3.in_features, 200)
    elif dataset == 'food101':
        model.l3 = nn.Linear(model.l3.in_features, 101)
    elif dataset == 'vireo172':
        model.l3 = nn.Linear(model.l3.in_features, 172)
    elif dataset == 'cifar100':
        model.l3 = nn.Linear(model.l3.in_features, 100)
    elif dataset == 'cifar10':
        model.l3 = nn.Linear(model.l3.in_features, 10)

    return model


def resnet34(dataset, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        print('done load model')
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        
    if dataset == 'tinyimagenet':
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif dataset == 'food101':
        model.fc = nn.Linear(model.fc.in_features, 101)
    elif dataset == 'viero172':
        model.fc = nn.Linear(model.fc.in_features, 172)
    elif dataset == 'cifar100':
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif dataset == 'cifar10':
        model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def resnet50(dataset, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        print('done load model')
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if dataset == 'tinyimagenet':
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif dataset == 'food101':
        model.fc = nn.Linear(model.fc.in_features, 101)
    elif dataset == 'viero172':
        model.fc = nn.Linear(model.fc.in_features, 172)
    elif dataset == 'cifar100':
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif dataset == 'cifar10':
        model.fc = nn.Linear(model.fc.in_features, 10)
    model_name = 'resnet50'
    return model


def resnet101(dataset, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if kwargs['num_classes'] != 1000 and pretrained:
        model = ResNet(Bottleneck, [3, 4, 23, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    else:
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if dataset == 'tinyimagenet':
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif dataset == 'food101':
        model.fc = nn.Linear(model.fc.in_features, 101)
    elif dataset == 'viero172':
        model.fc = nn.Linear(model.fc.in_features, 172)
    elif dataset == 'cifar100':
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif dataset == 'cifar10':
        model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def resnet152(dataset, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if kwargs['num_classes'] != 1000 and pretrained:
        model = ResNet(Bottleneck, [3, 8, 36, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    else:
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if dataset == 'tinyimagenet':
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif dataset == 'food101':
        model.fc = nn.Linear(model.fc.in_features, 101)
    elif dataset == 'viero172':
        model.fc = nn.Linear(model.fc.in_features, 172)
    elif dataset == 'cifar100':
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif dataset == 'cifar10':
        model.fc = nn.Linear(model.fc.in_features, 10)
    return model