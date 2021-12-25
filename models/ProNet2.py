
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from configs import model_configs

from AdasOptimizer.adasopt_pytorch import Adas

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        super(ResNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_channels = model_configs.num_channels
        self.layer1 = self.make_layer(block, model_configs.num_channels, layers[0])
        self.layer2 = self.make_layer(block, model_configs.num_channels, layers[1], 2)
        self.layer3 = self.make_layer(block, model_configs.num_channels, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2)
        self.max_pool = nn.MaxPool2d(2)
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride).to(self.device),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.max_pool(out)
        return out

class ProNet2(nn.Module):
    def __init__(self, num_classes=1):
        super(ProNet2, self).__init__()
        self.name = 'ProNet2'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print('Using GPU')
        self.input_shape = (64, 64)
        self.conv1 = nn.Conv2d(3, model_configs.num_channels>>4, 3, stride=1).to(self.device)
        self.conv2 = nn.Conv2d(model_configs.num_channels>>4, model_configs.num_channels>>2, 3, stride=1).to(self.device)
        self.conv3 = nn.Conv2d(model_configs.num_channels>>2, model_configs.num_channels, 3, stride=1).to(self.device)
        
        self.bn1 = nn.BatchNorm2d(model_configs.num_channels>>4).to(self.device)
        self.bn2 = nn.BatchNorm2d(model_configs.num_channels>>2).to(self.device)
        self.bn3 = nn.BatchNorm2d(model_configs.num_channels).to(self.device)
        
        self.avg_pool = nn.AvgPool2d(2).to(self.device)  
        self.max_pool = nn.MaxPool2d(2).to(self.device)
        self.resnet = ResNet(ResidualBlock, [2, 2, 2]).to(self.device)  
        
        self.fc1 = nn.Linear(1024 + 4, 256).to(self.device)
        self.fc_bn1 = nn.BatchNorm1d(256).to(self.device)

        self.fc2 = nn.Linear(256, 128).to(self.device)
        self.fc_bn2 = nn.BatchNorm1d(128).to(self.device)

        self.fc3 = nn.Linear(128, 64).to(self.device)
        self.fc_bn3 = nn.BatchNorm1d(64).to(self.device)
        
        self.fc4 = nn.Linear(64, 32).to(self.device)
        self.fc5 = nn.Linear(32, num_classes).to(self.device)
        
        self.train_losses = []
        
    def forward(self, x1, x2):
        # forward color features                               
        x1 = x1.view(-1, 3, self.input_shape[0], self.input_shape[1])  
        x2 = x2.view(-1, 4)
        x1 = self.max_pool(F.relu(self.bn1(self.conv1(x1))))  
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = self.avg_pool(F.relu(self.bn3(self.conv3(x1))) )
        x1 = F.relu(self.resnet(x1))       
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2] * x1.shape[3])           
        
        x = torch.cat((x1, x2), 1)                               
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=model_configs.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=model_configs.dropout, training=self.training)
        
        out = F.relu(self.fc_bn3(self.fc3(x)))
        out = self.fc4(out)  
        out = self.fc5(out)             
         
        return torch.sigmoid(out)
    
        
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
    
    def predict(self, input_1, input_2):
        input_1 = torch.FloatTensor(input_1).to(self.device).detach()
        input_2 = torch.FloatTensor(input_2).to(self.device).detach()  
        output = self.forward(input_1, input_2)
        return output.cpu().data.numpy()[0][0]
          
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