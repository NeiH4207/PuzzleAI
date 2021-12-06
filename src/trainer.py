import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from AdasOptimizer.adasopt_pytorch import Adas
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader, test_loader,
                 device=T.device("cpu"), lr=0.001, epochs=1000, batch_size=64,
                 print_every=1, save_every=100, save_dir="./trainned_models",
                 save_name="model.pt", verbose=True):
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.save_every = save_every
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose
        self.train_losses = []
        
        if optimizer != "sgd":
            self.optimizer = Adas(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []

    def split_batch(self, dataset, batch_size, shuffle=True):
        """
        Split dataset into batches
        :param dataset: dataset
        :param batch_size: batch size
        :return: batches
        """
        batches = []
        if shuffle:
            indices = np.random.permutation(len(dataset['data']))
            dataset['data'] = [dataset['data'][i] for i in indices]
            dataset['target'] = [dataset['target'][i] for i in indices]
        for i in range(0, len(dataset['data']), batch_size):
            batches.append((dataset['data'][i:i + batch_size], dataset['target'][i:i + batch_size]))
        return batches
    
    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            train_batches = self.split_batch(self.train_loader, self.batch_size)
            for batch_idx, (data, targets) in enumerate(train_batches):
                input_1, input_2 = [
                    [dt[0] for dt in data],
                    [dt[1] for dt in data]
                ]
                input_1 = Variable(T.FloatTensor(np.array(input_1).astype(np.float64)).to(self.device))
                input_2 = Variable(T.FloatTensor(np.array(input_2).astype(np.float64)).to(self.device))
                targets = T.FloatTensor(np.array(targets).reshape(-1, 1).astype(np.float64)).to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_1, input_2)
                loss = self.loss_v(output, targets)
                loss.backward()
                self.model.train_losses.append(loss.item())
                self.optimizer.step()

                if batch_idx % self.print_every == 0:
                    self.print_loss(epoch, batch_idx, loss.item())

                if batch_idx % self.save_every == 0:
                    # self.save_train_losses()
                    # self.test()
                    self.save_model(epoch, batch_idx)

    def loss_v(self, output, target):
        return F.mse_loss(output, target)
    
    def print_loss(self, epoch, batch_idx, loss):
        print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, loss))

    def load_model_from_path(self, path):
        self.model.load_state_dict(T.load(path))
    
    def save_train_losses(self):
        plt.plot(self.train_losses)
        plt.savefig("{}/{}_train_losses.png".format('output/', 'train_losses.png'))
    
    def save_model(self, epoch, batch_idx):
        T.save(self.model.state_dict(), "{}/{}_{}_{}.pt".format(self.save_dir, self.save_name, epoch, batch_idx))
        print("Model saved")
        
    def load_model(self, epoch, batch_idx):
        self.model.load_state_dict(T.load("{}/{}_{}_{}.pt".format(self.save_dir, self.save_name, epoch, batch_idx)))
        print("Model loaded")
        
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with T.no_grad():
            for data, _target in zip(self.test_loader['data'], self.test_loader['target']):
                input_1, input_2 = data[0], data[1]
                input_1 = T.FloatTensor(np.array(input_1).astype(np.float64)).to(self.device)
                input_2 = T.FloatTensor(np.array(input_2).astype(np.float64)).to(self.device)
                target = T.FloatTensor(np.array(_target).reshape(-1, 1).astype(np.float64)).to(self.device)
                output = self.model(input_1, input_2)
                loss = self.loss_v(output, target)
                target_out = np.round(output.cpu().numpy()[0][0])
                test_loss += loss.item()
                correct += 1 if target_out == _target else 0
        test_loss /= len(self.test_loader['data'])
        self.test_losses.append(test_loss)
        self.test_acc.append(correct / len(self.test_loader['data']))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader['data']),
            100. * correct / len(self.test_loader['data'])))