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
from torchsummary import summary

# from AdasOptimizer.adasopt_pytorch import Adas
# from configs import model_configs

cfg = {
    'VGG9': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(4 * cfg[vgg_name][-2] + 4, 32)
        self.fc2 = nn.Linear(32, 1)
        self.name = vgg_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print('Using GPU')
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
        x2 = x2.view(-1, 4)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, x2), 1)   
        out = self.fc1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out
    
    def predict(self, input_1, input_2):
        input_1 = torch.FloatTensor(input_1).to(self.device).detach()
        input_2 = torch.FloatTensor(input_2).to(self.device).detach() 
        input_1 = input_1.view(-1, input_1.shape[0], input_1.shape[1], input_1.shape[2])  
        input_2 = input_2.view(-1, 4)
        output = self.forward(input_1, input_2)
        return output.cpu().data.numpy()[0][0]
 
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()
        else:
            raise ValueError("Loss function not found")
        
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            self.optimizer = Adas(self.parameters(), lr=lr)
            
    def reset_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()
          
    def load_checkpoint(self, epoch, batch_idx):
        checkpoint = torch.load("{}/{}_{}_{}.pt".format(model_configs.save_dir, model_configs.save_name, epoch, batch_idx), map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.train_losses = checkpoint['train_loss']
        self.optimizer = checkpoint['optimizer']
        # self.load_state_dict(checkpoint)
        print('-- Load model succesfull!')
        
    def save_checkpoint(self, epoch, batch_idx):
        torch.save({
            'state_dict': self.state_dict(),
            'train_loss': self.train_losses,
            'optimizer': self.optimizer
        }, "{}/{}_{}_{}.pt".format(model_configs.save_dir, model_configs.save_name, epoch, batch_idx))
        
    def save_train_losses(self, train_losses):
        self.train_losses = train_losses
        
    def save(self, epoch, batch_idx):
        torch.save(self.state_dict(), "{}/{}_{}_{}.pt".format(model_configs.save_dir, model_configs.save_name, epoch, batch_idx))
        print("Model saved")
        
    def load(self, epoch, batch_idx):
        self.load_state_dict(torch.load("{}/{}_{}_{}.pt".format(model_configs.save_dir, model_configs.save_name, epoch, batch_idx)))
        print("Model loaded")

def test():
    summary(VGG('VGG9'), input_size=np.array([(3, 64, 64), (1, 4)]),
            batch_size=64, device='cpu')
    # net = SimpleDLA()
    # print(net)
    # x = torch.randn(1, 3, 32, 32)
    # y = net(x)
    # print(y.size())


if __name__ == '__main__':
    test()