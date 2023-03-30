# ==== Imports
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights

# ==== Convolutional model


class ConvModel(nn.Module):
    def __init__(self, dropout):

        super(ConvModel, self).__init__()
        # 2 cov lag blir opprette. Bildene har x, y og z verdi.
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=3, padding=0)

        # self.fc1 = nn.Linear(3*3*256, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.fc1 = nn.Linear(3*3*256, 1000)
        self.fc2 = nn.Linear(1000, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = F.relu(x)  # to activate function above

        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 3)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# ==== ResidualBlock(34)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, i_downsample=None):

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.i_downsample:
            residual = self.i_downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ==== Bottleneck(50,101)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identit\n,y
        x += identity
        x = self.relu(x)

        return x

# ==== Residual network model(34)


class ResNetModel(nn.Module):
    def __init__(self, block, layers, num_classes=2):

        super(ResNetModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        # self.avgPool = nn.AvgPool2d(7, stride=1)


        self.fc0 = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)

        return x

    # https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

# ==== Residual network model(50,101)


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(
            ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(
            ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(
            ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(512*ResBlock.expansion, 1000)
        self.fc1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.fc1(x)


        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes,
                      i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

# ==== Pretrained resnet34 imagenet model


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.my_new_layer = nn.Sequential(nn.Linear(1000, 2))
    
    def forward(self, x):
        x = self.model(x)
        x = self.my_new_layer(x)
        return x

# ==== Pretrained resnet50 imagenet model


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.my_new_layer = nn.Sequential(nn.Linear(1000, 2))
    
    def forward(self, x):
        x = self.model(x)
        x = self.my_new_layer(x)
        return x