import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from captcha_setting import *

'''torchvision内置ResNet，准确率只有73%'''
# class ResNet18(nn.Module):
#     def __init__(self):
#         super(ResNet18, self).__init__()
#         self.base = models.resnet18(pretrained=False)
#         self.base.fc = nn.Linear(self.base.fc.in_features,
#                                  MAX_CAPTCHA*ALL_CHAR_SET_LEN)
#
#     def forward(self, x):
#         out = self.base(x)
#         return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=64):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, MAX_CAPTCHA*ALL_CHAR_SET_LEN)
        # self.fc = nn.Sequential(
        #     nn.Linear((IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 512, 512),
        #     nn.Dropout(0.5),  # drop 50% of the neuron
        #     nn.LeakyReLU())
        self.rfc = nn.Sequential(nn.Linear(512, MAX_CAPTCHA * ALL_CHAR_SET_LEN))
        self.drop = nn.Dropout(0.5)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)                  # (batch_size, 512, 5, 15)
        out = nn.AdaptiveAvgPool2d(1)(out)      # (batch_size, 512, 1, 1)
        out = out.view(-1, 512)                 # (batch_size, 512)
        out = self.drop(out)
        out = self.rfc(out)                     # 248
        return out

def ResNet18():
    return ResNet(ResidualBlock)
