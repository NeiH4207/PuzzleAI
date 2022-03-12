import os
import cv2
import torch as T
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
from src.data_helper import DataProcessor
from tqdm import tqdm

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader=None, test_loader=None,
                 device=T.device("cpu"), lr=0.001, epochs=1000, batch_size=64,
                 n_repeats = 2, print_every=1, save_every=500, 
                 save_dir="./trainned_models",
                 save_name="model.pt", verbose=True):
        self.model = model
        self.model.set_loss_function(loss)
        self.model.set_optimizer(optimizer, lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_repeats = n_repeats
        self.print_every = print_every
        self.save_every = save_every
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose
        self.train_losses = []

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
            data_set = [dataset['data'][i] for i in indices]
            target_set = [dataset['target'][i] for i in indices]
        for i in range(0, len(data_set), batch_size):
            batches.append((data_set[i:i + batch_size], target_set[i:i + batch_size]))
        return batches
    
    def train(self):
        self.model.to(self.device)
        acc, val_loss = self.test()
        print("Accuracy: {}".format(acc))
        print("Validation loss: {}".format(val_loss))
        self.model.train()
        for iter in range(self.n_repeats):
            train_batches = self.split_batch(self.train_loader, self.batch_size)
            t = tqdm(train_batches, desc="Iter {}".format(iter))
            acc = 0
            val_loss = -1
            for batch_idx, (data, targets) in enumerate(t):
                input_1, input_2 = [
                    [DataProcessor.convert_image_to_three_dim(dt[0]) for dt in data],
                    [dt[1] for dt in data]
                ]
                input_1 = Variable(T.FloatTensor(np.array(input_1).astype(np.float64)).to(self.device), requires_grad=True)
                input_2 = Variable(T.FloatTensor(np.array(input_2).astype(np.float64)).to(self.device), requires_grad=True)
                targets = Variable(T.FloatTensor(np.array(targets).astype(np.float64)).to(self.device), requires_grad=True)
                self.model.reset_grad()
                output = self.model(input_1, input_2)
                loss = self.model.loss(output.flatten(), targets)
                loss.backward()
                self.train_losses.append(loss.item())
                self.model.step()
                
                t.set_postfix(loss=loss.item(), acc=acc, val_loss=val_loss)

                if batch_idx % self.save_every == 0 and batch_idx != 0 or batch_idx == len(train_batches) - 1:
                    self.save_train_losses()
                    self.model.save(iter, batch_idx)
                    acc, val_loss = self.test()

            
    def load_model_from_path(self, path):
        self.model.load_state_dict(T.load(path))
    
    def save_train_losses(self):
        plt.plot(self.train_losses)
        out_dir = 'output/train_losses'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig("{}/{}_{}".format(out_dir, self.model.name, 'train_losses.png'))
    
    def test(self):
        if len(self.test_loader['data']) == 0:
            print('Skipping test')
        self.model.eval()
        correct = 0
        total_loss = 0
        with T.no_grad():
            test_batches = self.split_batch(self.test_loader, len(self.test_loader['data']))
            for batch_idx, (data, targets) in enumerate(test_batches):
                input_1, input_2= [
                    [DataProcessor.convert_image_to_three_dim(dt[0]) for dt in data],
                    [dt[1] for dt in data]
                ]
                input_1 = Variable(T.FloatTensor(np.array(input_1).astype(np.float64)).to(self.device), requires_grad=True)
                input_2 = Variable(T.FloatTensor(np.array(input_2).astype(np.float64)).to(self.device), requires_grad=True)
                targets = Variable(T.FloatTensor(np.array(targets).astype(np.float64)).to(self.device), requires_grad=True)
                outputs = self.model(input_1, input_2).reshape(targets.shape)
                loss_value = self.model.loss(outputs.flatten(), targets)
                # correct if the difference less than 0.3
                correct += np.sum((outputs.flatten() - targets).abs().data.cpu().numpy() < 0.3)
                total_loss += np.sum(loss_value.item())
                
        self.valid_acc.append(correct / len(self.test_loader['data']))
        return self.valid_acc[-1], total_loss / len(test_batches)
