
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
args = dotdict({
    'lr': 0.001,
    'dropout': 0.5,
    'epochs': 20,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 64,
    'optimizer': 'adas',
})

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
        self.in_channels = args.num_channels
        self.layer1 = self.make_layer(block, args.num_channels, layers[0])
        self.layer2 = self.make_layer(block, args.num_channels, layers[1], 2)
        self.layer3 = self.make_layer(block, args.num_channels, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2)
        
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
        out = self.avg_pool(out)
        return out

class ProNet(nn.Module):
    def __init__(self, n_inputs, input_shape, num_classes=1):
        super(ProNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_inputs = n_inputs
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1).to(self.device)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1).to(self.device)
        self.conv3 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1).to(self.device)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1).to(self.device)
        self.conv5 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1).to(self.device)
        self.conv6 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1).to(self.device)
        
        self.bn1 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn2 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn3 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn4 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn5 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn6 = nn.BatchNorm2d(args.num_channels).to(self.device)
        
        self.resnet = ResNet(ResidualBlock, [2, 2, 2]).to(self.device)  
        
        self.last_channel_size = (int(args.num_channels) * input_shape[0] * input_shape[1]) >> 6
        # self.last_channel_size = args.num_channels * (self.board_x - 4) * (self.board_y - 4)
        self.fc1 = nn.Linear(self.last_channel_size*2+self.last_channel_size//4, 512).to(self.device)
        self.fc_bn1 = nn.BatchNorm1d(512).to(self.device)

        self.fc2 = nn.Linear(512, 128).to(self.device)
        self.fc_bn2 = nn.BatchNorm1d(128).to(self.device)

        self.fc3 = nn.Linear(128, num_classes).to(self.device)
        
        self.train_losses = []
        
    def forward(self, x1, x2, x3):
        #                                                           s: batch_size x n_inputs x board_x x board_y
        x1 = x1.view(-1, self.n_inputs, self.input_shape[0], self.input_shape[1])    
        x2 = x2.view(-1, self.n_inputs, self.input_shape[0], self.input_shape[1])
        x3 = x3.view(-1, self.n_inputs, self.input_shape[0], self.input_shape[1])
        x1 = F.relu(self.bn1(self.conv1(x1)))                          # batch_size x num_channels x board_x x board_y
        x1 = F.relu(self.bn2(self.conv2(x1)))                           # batch_size x num_channels x (board_x-4) x (board_y-4)
        x1 = F.relu(self.resnet(x1))        
        x1 = x1.view(-1, self.last_channel_size)                       # batch_size x last_channel_size
        # batch_size x num_channels x board_x x board_y
        x2 = F.relu(self.bn3(self.conv3(x2)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        x2 = F.relu(self.bn4(self.conv4(x2)))                # batch_size x num_channels x (board_x-4) x (board_y-4)
        x2 = F.relu(self.resnet(x2))
        x2 = x2.view(-1, self.last_channel_size)                       # batch_size x last_channel_size
        # batch_size x num_channels x board_x x board_y
        x3 = F.relu(self.bn5(self.conv5(x3)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        x3 = F.relu(self.bn6(self.conv6(x3)))                # batch_size x num_channels x (board_x-4) x (board_y-4)
        x3 = F.relu(self.resnet(x3))
        x3 = x3.view(-1, self.last_channel_size//4)  
        
        x = torch.cat((x1, x2, x3), 1)                                # batch_size x 3*last_channel_size
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=args.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=args.dropout, training=self.training)
        out = self.fc3(x)                        
        return out
          
          
    def save_checkpoint(self, folder='trainned_models', filename='model.pt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses,
        }, filepath)
        
        
    def load_checkpoint(self, folder='trainned_models', filename='model.pt'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename) 
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_loss']
        # self.load_state_dict(checkpoint)
        print('-- Load model succesfull!')