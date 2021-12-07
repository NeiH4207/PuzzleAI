import csv
import _pickle as cPickle
import os
import numpy as np
import cv2
from numpy.core.numeric import indices
from configs import *
from copy import deepcopy as copy
import h5py

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
    
    def load_data_from_binary_file(self, path, file_name):
        """
        Load data from binary file
        :param path: path to the file
        :return: data
        """
        file_dir = path + file_name
        dataset = self.unpickle(file_dir)
        return dataset
    
    def load_data(self, path, file_name):

        file_dir = path + file_name
        dataset = h5py.File(file_dir, 'r')
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
        
        index_img_blocks = []
        block_length = block_dim[0] * block_dim[1]
        for index in range(0, block_length):
            index_blocks = [np.zeros(block_size) if i != index else np.ones(block_size) 
                            for i in range(block_length)]
            index_image = self.merge_blocks(index_blocks, block_dim)
            index_img_blocks.append(index_image)
        
        for i in range(len(dataset)):
            image = dataset[i]
            image = self.convert_array_to_rgb_image(image, 32, 32)
            image = cv2.resize(image, IMG_SIZE, interpolation = cv2.INTER_AREA)
            blocks = self.split_image_to_blocks(image, block_dim)
            dropped_blocks, lost_blocks, lost_labels = self.random_drop_blocks(blocks)
            block_shape = blocks[0].shape
            dropped_index_img_blocks = [np.ones(block_size) if i in lost_labels else np.zeros(block_size)
                                        for i in range(block_length)]
            # 
            
            for index in range(1, block_length):
                if index in lost_labels:
                    continue
                recovered_image_blocks = copy(dropped_blocks)
                recovered_image = self.merge_blocks(recovered_image_blocks, block_dim, mode=1)
                cp_dropped_index_img_blocks = copy(dropped_index_img_blocks)
                cp_dropped_index_img_blocks[index] = np.ones(block_size)
                dropped_index_image = self.merge_blocks(cp_dropped_index_img_blocks, block_dim) 
                data = [recovered_image, index_img_blocks[index], dropped_index_image]
                new_dataset['data'].append(data)
                new_dataset['target'].append(1)
                
                angle = np.random.randint(1, 4)
                rotate_block = np.rot90(recovered_image_blocks[index], angle)
                recovered_image_blocks[index] = rotate_block
                recovered_image = self.merge_blocks(recovered_image_blocks, block_dim, mode=1)
                data = [recovered_image, index_img_blocks[index], dropped_index_image]
                new_dataset['data'].append(data)
                new_dataset['target'].append(0)
                
                
            dropped_index_image = self.merge_blocks(dropped_index_img_blocks, block_dim) 
            ''' false blocks '''
            for index in lost_labels:
                for false_index in lost_labels:
                    if index == false_index:
                        continue
                    for angle in range(0, 4):
                        if np.random.uniform() > 0.5:
                            continue
                        recovered_image_blocks = copy(dropped_blocks)
                        recovered_image_blocks[index] = np.rot90(blocks[false_index])
                        recovered_image = self.merge_blocks(recovered_image_blocks, block_dim, mode=1)
                        data = [recovered_image, index_img_blocks[index], dropped_index_image]
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
            if np.random.uniform() < train_size:
                train_dataset_indices.append(i)
            else:
                test_dataset_indices.append(i)
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
        train_dataset = self.load_data_from_binary_file(file_dir, 'train_dataset.bin')
        test_dataset = self.load_data_from_binary_file(file_dir, 'test_dataset.bin')
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
    
    def convert_image_to_three_dim(self, image):
        """
        Convert image to three dim
        :param image: image
        :return: three dim image
        """
        red_image = image[:, :, 0]
        green_image = image[:, :, 1]
        blue_image = image[:, :, 2]
        three_dim_image = np.stack((red_image, green_image, blue_image), axis=0)
        return three_dim_image
    
    def convert_three_dim_to_image(self, three_dim_image):
        """
        Convert three dim image to image
        :param three_dim_image: three dim image
        :return: image
        """
        image = three_dim_image[:, :, 0]
        return image
    
    def convert_rgb_to_bw(self, rgb_image):
        """
        Convert rgb image to black white image
        :param rgb_image: rgb image
        :return: black white image
        """
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def split_image_to_blocks(self, image, block_dim=(2, 2)):
        """
        Divide image to blocks
        :param image: image
        :param block_dim: dimension of block
        :return: blocks
        """
        block_size = image.shape[0] // block_dim[0], image.shape[1] // block_dim[1]
        blocks = []
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                block = image[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]]
                blocks.append(block)
        return blocks
    
    def merge_blocks(self, blocks, block_dim=(2, 2), mode=0):
        """
        Recover image from blocks
        :param blocks: blocks
        :param n_rows: number of rows
        :param n_cols: number of columns
        :param block_size: size of block
        :mode: 0: black white, 1: rgb
        :return: image
        """
        n_rows, n_cols = block_dim
        image_size = blocks[0].shape[0] * n_rows, blocks[0].shape[1] * n_cols
        h, w = image_size
        block_size = blocks[0].shape[0], blocks[0].shape[1]
        image_shape = (h, w) if mode == 0 else (h, w, 3)
        image = np.zeros(image_shape, dtype=np.uint8)
        for i in range(n_rows):
            for j in range(n_cols):
                block = blocks[i * n_cols + j]
                image[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]] = block
        return image
        
        
    def shuffle_blocks(self, blocks, rotate=False):
        """
        Shuffle blocks
        :param blocks: blocks
        :return: shuffled blocks and labels each block
        """
        n_blocks = len(blocks)
        shuffled_blocks = [blocks[0]]
        shuffled_labels = [0]
        indices = set(range(1, n_blocks))
        while len(indices) > 0:
            index = np.random.choice(list(indices))
            indices.remove(index)
            block = blocks[index]
            if rotate:
                block = np.rot90(block, k=np.random.randint(4))
            shuffled_blocks.append(block)
            shuffled_labels.append(index)
        return np.array(shuffled_blocks), np.array(shuffled_labels)
    
    def random_drop_blocks(self, blocks, prob=None):
        """
        Random drop blocks
        :param blocks: blocks
        :param prob: probability of drop
        :return: dropped blocks
        """
        if prob is None:
            prob = max(np.random.uniform(), 0.1)
        block_size = blocks[0].shape
        n_blocks = len(blocks)
        dropped_blocks = []
        lost_blocks = []
        lost_list = []
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
    
DataProcessor = DataHelper()