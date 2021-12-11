import csv
import _pickle as cPickle
import os
import numpy as np
import cv2
from numpy.random.mtrand import random
from configs import img_configs
from copy import deepcopy as copy
import pandas as pd
import urllib
from tqdm import tqdm

class DataHelper:
    def __init__(self):
        pass
    
    def downloadImage(self, url):
        """
        Download image from url
        :param url: url of image
        :return: image
        """
        try:
            image = urllib.request.urlopen(url)
            image = np.asarray(bytearray(image.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        except:
            print('Error: ' + url)
    
    def load_data_from_binary_file(self, path, file_name):
        """
        Load data from binary file
        :param path: path to the file
        :return: data
        """
        file_dir = path + file_name
        dataset = self.unpickle(file_dir)
        return dataset
    
    def save_data_to_binary_file(self, dataset, path):
        """
        Save dataset to binary file
        :param dataset: dataset
        :param path: path to the file
        :return: None
        """
        dir_path = path.split('/')[:-1]
        dir_path = '/'.join(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, 'wb') as f:
            cPickle.dump(dataset, f)
        
        
    def read_url_csv(self, file_dir, file_name, chunksize=256, skiprows=0):
        """
        crawl image from url in csv file
        :param file_dir: directory of file
        :param file_name: file name
        :param low_memory: low memory
        :return: dataframe
        """
        file_path = file_dir + file_name
        reader = pd.read_csv(file_path, sep=',',
                             chunksize=chunksize,
                             skiprows=skiprows,
                             low_memory=True)
        
        # read url from OriginalLandingURL column
        for index, chunk in enumerate(reader):
            data = []
            t = tqdm(chunk['OriginalURL'], desc='Reading url')
            for url in t:
                image = self.downloadImage(url)
                if image is None:
                    continue
                # resize image if valid size
                try:
                    image = cv2.resize(image, img_configs['max-size'])
                    data.append(image)
                except:
                    continue
            self.save_data_to_binary_file(data, file_dir + 'images/image_data_batch_' + str(index) + '.bin')
            
            size = img_configs['max-size'][0] >> 1 # np.random.randint(1, 3)
            block_size = (size, size)
            block_dim = (img_configs['max-size'][0] // size, img_configs['max-size'][1] // size)
            data_batch = self.generate_data(data, block_dim, block_size, is_array=False)
            self.save_data_to_binary_file(data_batch, file_dir + 'data_batch_' + str(index) + '.bin')
            print('Finish saving data batch ' + str(index) + ' / num rows: ' + str(len(data_batch['data'])))
        return data
            
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def generate_data(self, dataset, block_dim, block_size, is_array=True):
        HEIGHT = block_size[0] * block_dim[0]
        WIDTH = block_size[1] * block_dim[1]
        IMG_SIZE = (HEIGHT, WIDTH)
        
        new_dataset = {
            'data': [],
            'target': [],
            'block_dim': [],
            'block_size': [],
            'image_size': []
        }
        
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        
        index_imgs = np.zeros((block_dim[0], block_dim[1], HEIGHT, WIDTH), dtype=np.int8)
        
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                index_imgs[i][j][i * block_size[0]:(i + 1) * block_size[0], 
                                 j * block_size[1]:(j + 1) * block_size[1]] = np.ones((block_size[0], block_size[1]), dtype=np.int8)
        
        for org_image in tqdm(dataset['data'], desc='Generating data'):
            # org_image = dataset[i]
            if is_array:
                org_image = self.convert_array_to_rgb_image(org_image, 32, 32)
            org_image = cv2.resize(org_image, IMG_SIZE, interpolation = cv2.INTER_AREA)
            # rotate image
            image = np.rot90(org_image, k=np.random.randint(0, 4))
            # cv2.imwrite('output/sample.png', image)
        
            blocks = self.split_image_to_blocks(image, block_dim)
            dropped_blocks, lost_block_labels, _ = self.random_drop_blocks(blocks)
            lost_index_img_blocks = np.empty((block_dim[0], block_dim[1], block_size[0], block_size[1]), dtype=np.int8)
            
            for x in range(block_dim[0]):
                for y in range(block_dim[1]):
                    if lost_block_labels[x][y] == 0:
                        lost_index_img_blocks[x][y] = np.zeros((block_size[0], block_size[1]), dtype=np.int8)
                    else:
                        lost_index_img_blocks[x][y] = np.ones((block_size[0], block_size[1]), dtype=np.int8)
            
            lost_positions = set()
            for i in range(block_dim[0]):
                for j in range(block_dim[1]):
                    if lost_block_labels[i][j] == 1:
                        lost_positions.add((i, j))
                        
            block_shape = blocks[0][0].shape
            
            if len(list(lost_positions)) < block_dim[0] * block_dim[1] - 1:
                for x, y in [(i, j) for i in range(block_dim[0]) for j in range(block_dim[1])]:
                    if (x, y) in lost_positions:
                        continue
                    recovered_image_blocks = copy(dropped_blocks)
                    recovered_image = self.merge_blocks(recovered_image_blocks)
                    cp_dropped_index_img_blocks = copy(lost_index_img_blocks)
                    cp_dropped_index_img_blocks[x][y] = np.ones(block_size)
                    dropped_index_image = self.merge_blocks(cp_dropped_index_img_blocks, mode='gray') 
                    data = [recovered_image, index_imgs[x][y], dropped_index_image]
                    # cv2.imwrite('output/sample.png', recovered_image)
                    new_dataset['data'].append(data)
                    new_dataset['target'].append((1, 0))
                    new_dataset['block_dim'].append(block_dim)
                    new_dataset['block_size'].append(block_size)
                    new_dataset['image_size'].append(IMG_SIZE)
                    
                    angle = np.random.randint(1, 4)
                    rotated_block = np.rot90(recovered_image_blocks[x][y], angle)
                    recovered_image_blocks[x][y] = rotated_block
                    recovered_image = self.merge_blocks(recovered_image_blocks)
                    data = [recovered_image, index_imgs[x][y], dropped_index_image]
                    # cv2.imwrite('output/sample.png', recovered_image)
                    new_dataset['data'].append(data)
                    new_dataset['target'].append((0, 4 - angle))
                    new_dataset['block_dim'].append(block_dim)
                    new_dataset['block_size'].append(block_shape)
                    new_dataset['image_size'].append(IMG_SIZE)
                
                
            dropped_index_image = self.merge_blocks(lost_index_img_blocks, mode='gray') 
            ''' false blocks '''
            for x1, y1 in lost_positions:
                for x2, y2 in lost_positions:
                    if x1 == x2 and y1 == y2:
                        continue
                    chosen = False
                    for i in range(4):
                        x, y = x1 + dx[i], y1 + dy[i]
                        if x < 0 or x >= block_dim[0] or y < 0 or y >= block_dim[1]:
                            continue
                        if (x, y) not in lost_positions:
                            chosen = True
                            break
                    if not chosen:
                        continue
                    angle = np.random.randint(0, 4)
                    recovered_image_blocks = copy(dropped_blocks)
                    recovered_image_blocks[x1][y1] = np.rot90(blocks[x2][y2], k=angle)
                    recovered_image = self.merge_blocks(recovered_image_blocks, mode='rgb')
                    data = [recovered_image, index_imgs[x1][y1], dropped_index_image]
                    # cv2.imwrite('output/sample.png', recovered_image)
                    new_dataset['data'].append(data)
                    new_dataset['target'].append((0, -1))
                    new_dataset['block_dim'].append(block_dim)
                    new_dataset['block_size'].append(block_shape)
                    new_dataset['image_size'].append(IMG_SIZE)
                    
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
            'target': [data['target'][i] for i in train_dataset_indices],
            'block_dim': [data['block_dim'][i] for i in train_dataset_indices],
            'block_size': [data['block_size'][i] for i in train_dataset_indices],
            'image_size': [data['image_size'][i] for i in train_dataset_indices]
        }
        test_dataset = {
            'data': [data['data'][i] for i in test_dataset_indices],
            'target': [data['target'][i] for i in test_dataset_indices],
            'block_dim': [data['block_dim'][i] for i in test_dataset_indices],
            'block_size': [data['block_size'][i] for i in test_dataset_indices],
            'image_size': [data['image_size'][i] for i in test_dataset_indices]
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
    
    def convert_rgb_to_gray(self, rgb_image):
        """
        Convert rgb image to gray image
        :param rgb_image: rgb image
        :return: gray image
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
            row = []
            for j in range(block_dim[1]):
                block = image[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]]
                row.append(block)
            blocks.append(row)
        return np.array(blocks)
    
    def merge_blocks(self, blocks, mode='rgb'):
        """
        Recover image from blocks
        :param blocks: blocks in image
        :param mode: rgb or gray
        :return: image
        """
        n_rows, n_cols = blocks.shape[0], blocks.shape[1]
        image_size = blocks[0][0].shape[0] * n_rows, blocks[0][0].shape[1] * n_cols
        h, w = image_size
        block_size = blocks[0][0].shape[0], blocks[0][0].shape[1]
        image_shape = (h, w) if mode == 'gray' else (h, w, 3)
        image = np.empty(image_shape, dtype=np.uint8)
        for i in range(n_rows):
            for j in range(n_cols):
                image[i * block_size[0]:(i + 1) * block_size[0], 
                      j * block_size[1]:(j + 1) * block_size[1]] = blocks[i][j]
        return image
        
        
    def shuffle_blocks(self, blocks, rotate=False):
        """
        Shuffle blocks
        :param blocks: blocks
        :return: shuffled blocks and labels
        """
        n_rows, n_cols = blocks.shape[0], blocks.shape[1]
        shuffled_labels = np.zeros((n_rows, n_cols))
        shuffled_blocks = np.empty(blocks.shape, dtype=np.ndarray)
        indices = set((i, j) for i in range(n_rows) for j in range(n_cols) if (i, j) != (0, 0))
        for i in range(n_rows):
            for j in range(n_cols):
                if (i, j) != (0, 0):
                    x, y = np.random.choice(list(indices))
                    indices.remove((x, y))
                    block = blocks[x][y]
                    if rotate:
                        block = np.rot90(block, k=np.random.randint(4))
                    shuffled_blocks[x][y] = block
                    shuffled_labels[x][y] = i * n_cols + j
        return shuffled_blocks, shuffled_labels
    
    def random_drop_blocks(self, blocks, prob=None):
        """
        Random drop blocks
        :param blocks: blocks
        :param prob: probability of drop
        :return: dropped blocks
        """
        if prob is None:
            prob = np.random.uniform()
        n_rows, n_cols = blocks.shape[0], blocks.shape[1]
        n_steps = prob * n_rows * n_cols
        masked = np.zeros((n_rows, n_cols), dtype=np.uint8)
        dx = [0, 1, 0, -1]
        dy = [-1, 0, 1, 0]
        
        block_size = blocks[0][0].shape
        dropped_blocks = np.empty(blocks.shape, dtype=np.ndarray)
        lost_block_labels = np.zeros((n_rows, n_cols))
        next_position = set()
        x, y = np.random.randint(0, n_rows), np.random.randint(0, n_cols)
        masked[x][y] = 1
        for i in range(4):
            _x, _y = x + dx[i], y + dy[i]
            if 0 <= _x < n_rows and 0 <= _y < n_cols:
                next_position.add((_x, _y))
        
        for i in range(int(n_steps)):
            next_pos = list(next_position)[np.random.choice(len(next_position))]
            x = next_pos[0]
            y = next_pos[1]
            masked[x][y] = 1
            next_position.remove(next_pos)
            for j in range(4):
                _x = x + dx[j]
                _y = y + dy[j]
                if _x >= 0 and _x < n_rows and _y >= 0 and _y < n_cols \
                    and (x, y) not in next_position and masked[_x][_y] == 0:
                    next_position.add((_x, _y))
            
        for i in range(n_rows):
            for j in range(n_cols):
                if masked[i][j] == 0:
                    dropped_blocks[i][j] = np.zeros(block_size)
                    lost_block_labels[i][j] = 1
                else:
                    dropped_blocks[i][j] = blocks[i][j]
        return dropped_blocks, lost_block_labels, masked
    
    def drop_all_blocks(self, blocks, dim=(2,2)):
        """
        Drop all blocks
        :param blocks: blocks
        :param dim: dimension of block
        :return: dropped blocks and labels
        """
        n_rows, n_cols = blocks.shape[0], blocks.shape[1]
        dropped_blocks = np.empty(blocks.shape, dtype=np.ndarray)
        dropped_labels = np.zeros((n_rows, n_cols))
        masked = np.zeros((n_rows, n_cols), dtype=np.uint8)
        for i in range(n_rows):
            for j in range(n_cols):
                if (i, j) != (0, 0):
                    dropped_blocks[i][j] = np.zeros(blocks[0][0].shape)
                    dropped_labels[i][j] = 1
                else:
                    masked[i][j] = 1
                    dropped_blocks[i][j] = blocks[i][j]
        return dropped_blocks, dropped_labels, masked
    
DataProcessor = DataHelper()