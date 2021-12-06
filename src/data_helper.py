import csv
import _pickle as cPickle
import os
import numpy as np
import cv2
from numpy.core.numeric import indices
from configs import *
from copy import deepcopy as copy

# Data for computer vision
class DataHelper:
    def __init__(self):
        pass

    def get_data(self, path):
        """
        Get data from a file
        :param path: path to the file
        :return: data
        """
        with open(path, 'r') as f:
            data = f.read()
        return data
    
    def load_data_from_binary_file(self, path):
        """
        Load data from binary file
        :param path: path to the file
        :return: data
        """
        dataset = self.unpickle(path)
        return dataset
    def save_data_to_binary_file(self, dataset, path):
        """
        Save dataset to binary file
        :param dataset: dataset
        :param path: path to the file
        :return: None
        """
        with open(path, 'wb') as f:
            cPickle.dump(dataset, f)
        
        
    def get_data_from_csv(self, path):
        """
        Get data from a csv file
        :param path: path to the file
        :return: data
        """
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data
    
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def transform(self, dataset, block_dim, block_size):
        new_dataset = {
            'data': [],
            'target': []
        }
        HEIGHT = img_configs['block-size'][0] * img_configs['block-dim'][0]
        WIDTH = img_configs['block-size'][1] * img_configs['block-dim'][1]
        IMG_SIZE = (HEIGHT, WIDTH)
        
        
        for i in range(len(dataset)):
            image = dataset[i]
            image = self.convert_array_to_rgb_image(image, 32, 32)
            image = self.convert_rgb_to_bw(image)
            image = cv2.resize(image, IMG_SIZE, interpolation = cv2.INTER_AREA)
            blocks = self.split_image_into_blocks(image, block_size)
            dropped_blocks, lost_blocks, lost_labels = self.random_drop_blocks(blocks)
            index_img_blocks = []
            for index in range(0, len(blocks)):
                index_blocks = [np.zeros(block_size) if i != index else np.ones(block_size) 
                                for i in range(len(blocks))]
                index_image = self.recover_image_from_blocks(index_blocks, block_dim, block_size)
                index_img_blocks.append(index_image)
            
            ''' true blocks '''
            for index in range(1, len(blocks)):
                if index not in lost_labels:
                    new_dataset['data'].append([image, index_img_blocks[index]])
                    new_dataset['target'].append(1)
                    
            ''' false blocks '''
            for index in lost_labels:
                for false_index in lost_labels:
                    if false_index == index:
                        continue
                    recovered_image_blocks = copy(dropped_blocks)
                    recovered_image_blocks[index] = blocks[false_index]
                    recovered_image = self.recover_image_from_blocks(recovered_image_blocks, block_dim, block_size)
                    data = [recovered_image, index_img_blocks[index]]
                    new_dataset['data'].append(data)
                    new_dataset['target'].append(0)
                    
        return new_dataset
    
    def split_dataset(self, data:dict, train_size=0.8, test_size=0.2, 
                      saved=False, shuffle = True, file_dir='input/'):
        """
        Split dataset into train and test
        :param data: dataset
        :param train_size: size of train
        :param test_size: size of test
        :param saved: save the dataset
        :param shuffle: shuffle the dataset
        :param file_dir: directory to save the dataset
        :return: train, test
        """
        
        train_dataset_indices = []
        test_dataset_indices = []
        for i in range(len(data['data'])):
            if np.random.uniform() < test_size:
                test_dataset_indices.append(i)
            else:
                train_dataset_indices.append(i)
        if shuffle:
            np.random.shuffle(train_dataset_indices)
            np.random.shuffle(test_dataset_indices)
        train_dataset = {
            'data': [data['data'][i] for i in train_dataset_indices],
            'target': [data['target'][i] for i in train_dataset_indices]
        }
        test_dataset = {
            'data': [data['data'][i] for i in test_dataset_indices],
            'target': [data['target'][i] for i in test_dataset_indices]
        }
        if saved:
            self.save_data_to_binary_file(train_dataset, file_dir + 'train_dataset.bin')
            self.save_data_to_binary_file(test_dataset, file_dir + 'test_dataset.bin')
            
        return train_dataset, test_dataset
    
    
    def load_train_test_dataset(self, file_dir, shuffle=True):
        """
        Load train and test dataset
        :param file_dir: directory of dataset
        :return: train and test dataset
        """
        train_dataset = self.load_data_from_binary_file(file_dir + 'train_dataset.bin')
        test_dataset = self.load_data_from_binary_file(file_dir + 'test_dataset.bin')
        if shuffle:
            indices = np.arange(len(train_dataset['data']))
            np.random.shuffle(indices)
            train_dataset['data'] = [train_dataset['data'][i] for i in indices]
            train_dataset['target'] = [train_dataset['target'][i] for i in indices]
        return train_dataset, test_dataset
    
    def convert_array_to_rgb_image(self, array, height, width):
        """
        Convert array to red blue green image
        :param array: flatten array
        :param height: height of image
        :param width: width of image
        :return: rgb image
        """
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = np.array(array[0:height * width]).reshape(height, width)
        rgb_image[:, :, 1] = np.array(array[height * width:height * width * 2]).reshape(height, width)
        rgb_image[:, :, 2] = np.array(array[height * width * 2:height * width * 3]).reshape(height, width)
        return rgb_image
    
    def convert_rgb_to_bw(self, rgb_image):
        """
        Convert rgb image to black white image
        :param rgb_image: rgb image
        :return: black white image
        """
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def split_image_into_blocks(self, image, block_size=(32, 32)):
        """
        Divide image to blocks
        :param image: image
        :param block_size: size of block
        :return: blocks
        """
        height, width = image.shape
        blocks = []
        for i in range(0, height, block_size[0]):
            for j in range(0, width, block_size[1]):
                block = image[i:i + block_size[0], j:j + block_size[1]]
                blocks.append(block)
        return blocks
    
    def recover_image_from_blocks(self, blocks, block_dim=(2, 2), block_size=(128, 128)):
        """
        Recover image from blocks
        :param blocks: blocks
        :param n_rows: number of rows
        :param n_cols: number of columns
        :param block_size: size of block
        :return: image
        """
        n_rows, n_cols = block_dim
        h, w = block_size  
        image = np.zeros((h * n_rows, w * n_cols), dtype=np.uint8)
        for i in range(n_rows):
            for j in range(n_cols):
                image[i * h:(i + 1) * h, j * w:(j + 1) * w] = blocks[i * n_cols + j]
        return np.array(image)
        
        
    def shuffle_blocks(self, blocks):
        """
        Shuffle blocks
        :param blocks: blocks
        :return: shuffled blocks and labels each block
        """
        n_blocks = len(blocks)
        shuffled_blocks = []
        shuffled_labels = []
        indices = set(range(n_blocks))
        while len(indices) > 0:
            index = np.random.choice(list(indices))
            indices.remove(index)
            shuffled_blocks.append(blocks[index])
            shuffled_labels.append(index)
        return shuffled_blocks, shuffled_labels
    
    def random_drop_blocks(self, blocks):
        """
        Random drop blocks
        :param blocks: blocks
        :param prob: probability of drop
        :return: dropped blocks
        """
        prob = max(np.random.uniform(), 0.1)
        block_size = blocks[0].shape
        n_blocks = len(blocks)
        dropped_blocks = []
        lost_blocks = []
        lost_list = []
        while len(lost_list) == 0:
            for i in range(1, n_blocks):
                if np.random.uniform() < prob:
                    lost_list.append(i)
                
        for i in range(0, n_blocks):
            if i in lost_list:
                dropped_blocks.append(np.zeros(block_size))
                lost_blocks.append(blocks[i])
            else:
                dropped_blocks.append(blocks[i])
        return dropped_blocks, lost_blocks, lost_list