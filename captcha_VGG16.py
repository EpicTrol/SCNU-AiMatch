import torch.nn as nn
# import setting as st
from captcha_setting import *

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        '''输入改为3，因为RGB3通道，原来是1'''

        self.layer1 = self.make_layers([64, 64, 'M'], 1, droup_out=False)
        self.layer2 = self.make_layers([128, 128, 'M'], 64, droup_out=True)
        self.layer3 = self.make_layers(
            [256, 256, 256, 'M', 512, 512, 512, 'M'], 128, droup_out=False)
        self.layer4 = self.make_layers(
            [512, 512, 512, 'M'], 512, droup_out=True)
        self.fc = nn.Sequential(
            nn.Linear((IMAGE_WIDTH // 32) * (IMAGE_HEIGHT // 32) * 512, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.LeakyReLU())
        self.rfc = nn.Sequential(nn.Linear(1024, MAX_CAPTCHA*ALL_CHAR_SET_LEN))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1) #-1, out.size(0)
        out = self.fc(out)
        out = self.rfc(out)
        return out

    def make_layers(self, cfg, inc, droup_out=False):
        layers = []
        i = inc
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(i, v, kernel_size=3, padding=1)
                if droup_out:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.Dropout(0.5), nn.LeakyReLU()]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU()]
                i = v
        return nn.Sequential(*layers)

    def model_name(self):
        return self.__class__.__name__
    
