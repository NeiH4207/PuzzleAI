'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
args = dotdict({
    'lr': 0.001,
    'dropout': 0.7,
    'epochs': 20,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 64,
    'optimizer': 'adas',
    'save_dir': './trainned_models',
    'save_name': 'model_2x2'
})


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DLA(nn.Module):
    def __init__(self, input_shape=(32,32), block=BasicBlock, num_classes=1):
        super(DLA, self).__init__()
        
        self.input_shape = input_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = 'dla'
        
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)
        
        self.train_losses = []

    def forward(self, x1, x2, x3):
        x1 = x1.view(-1, 1, self.input_shape[0], self.input_shape[1])    
        x2 = x2.view(-1, 1, self.input_shape[0], self.input_shape[1])
        x3 = x3.view(-1, 1, self.input_shape[0], self.input_shape[1])
        x = torch.cat((x1, x2, x3), 1)
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)    
        return torch.sigmoid(out)
    
    def predict(self, input_1, input_2, input_3):
        input_1 = torch.FloatTensor(input_1).float().to(self.device)
        input_2 = torch.FloatTensor(input_2).float().to(self.device)
        input_3 = torch.FloatTensor(input_3).float().to(self.device)
        return self.forward(input_1, input_2, input_3).cpu().data.numpy()[0][0]
          
    def save_checkpoint(self, epoch, batch_idx):
        torch.save({
            'state_dict': self.state_dict(),
            'train_loss': self.train_losses,
        }, "{}/{}_{}_{}.pt".format(args.save_dir, args.save_name, epoch, batch_idx))
        
        
    def load_checkpoint(self, epoch, batch_idx):
        checkpoint = torch.load("{}/{}_{}_{}.pt".format(args.save_dir, args.save_name, epoch, batch_idx), map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.train_losses = checkpoint['train_loss']
        # self.load_state_dict(checkpoint)
        print('-- Load model succesfull!')
    
    def save_train_losses(self, train_losses):
        self.train_losses = train_losses
        
    def save(self, epoch, batch_idx):
        torch.save(self.state_dict(), "{}/{}_{}_{}.pt".format(args.save_dir, args.save_name, epoch, batch_idx))
        print("Model saved")
        
    def load(self, epoch, batch_idx):
        self.load_state_dict(torch.load("{}/{}_{}_{}.pt".format(args.save_dir, args.save_name, epoch, batch_idx)))
        print("Model loaded")


def test():
    net = DLA()
    print(net)
    x1 = torch.randn(1, 3, 32, 32)
    x2 = torch.randn(1, 3, 32, 32)
    x3 = torch.randn(1, 3, 32, 32)
    y = net(x1, x2, x3)
    print(y.size())


if __name__ == '__main__':
    test()
