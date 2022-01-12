'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import numpy as np
import torch
from torch import optim
import torch.nn as nn

from AdasOptimizer.adasopt_pytorch import Adas
from configs import model_configs


class Pronet(nn.Module):
    def __init__(self):
        self.name = 'ProNet'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print('Using GPU')
        else:
            print('Using CPU')
        self.train_losses = []
        
    def forward(self, x1, x2):
        pass


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
        }, "{}/{}_{}_{}.pt".format(model_configs.save_dir, self.name, epoch, batch_idx))
        
    def save_train_losses(self, train_losses):
        self.train_losses = train_losses
        
    def save(self, epoch, batch_idx):
        torch.save(self.state_dict(), "{}/{}_{}_{}.pt".format(model_configs.save_dir, self.name, epoch, batch_idx))
        print("Model saved")
        
    def load(self, epoch, batch_idx):
        self.load_state_dict(torch.load("{}/{}_{}_{}.pt".format(model_configs.save_dir, self.name, epoch, batch_idx)))
        print("Model loaded")
