'''Simplified version of DLA in PyTorch.
Note this implementation is not identical to the original paper version.
But it seems works fine.
See dla.py for the original paper version.
Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary.torchsummary import summary
from models.pronet import Pronet
cfg = {
    'VGG7': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG9': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Pronet):
    def __init__(self, name, using_gpu=False):
        super(Pronet, self).__init__()
        self.features = self._make_layers(cfg[name])
        self.fc1 = nn.Linear(4 * cfg[name][-2] + 8, 32)
        self.fc2 = nn.Linear(32, 1)
        self.name = name
        self.name = 'ProNet'
        self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'
        if self.device == 'cuda':
            print('Using GPU')
        else:
            print('Using CPU')
        self.train_losses = []

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # forward color features  
        out = self.features(x1)
        x2 = x2.view(-1, 8)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, x2), 1)   
        out = self.fc1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out
    
    # def forward(self, x1):
    #     # forward color features  
    #     out = self.features(x1)
    #     out = out.view(out.size(0), -1)
    #     out = self.fc1(out)
    #     out = self.fc2(out)
    #     out = torch.sigmoid(out)
    #     return out
    
def test():
    summary(VGG('VGG9'), input_size=(3, 64, 64),
            batch_size=64, device='cpu')
    # net = SimpleDLA()
    # print(net)
    # x = torch.randn(1, 3, 32, 32)
    # y = net(x)
    # print(y.size())


if __name__ == '__main__':
    test()