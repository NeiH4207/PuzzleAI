import csv
import _pickle as cPickle
import os
import numpy as np
import cv2
from multiprocessing import Pool
import psutil
from torch.nn import parameter
from configs import img_configs
from copy import deepcopy as copy
import pandas as pd
import urllib
from tqdm import tqdm
from random import random, randint, SystemRandom

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
        
        
    def read_url_csv(self, file_dir, file_name, chunksize=256, skiprows=0, first_batch=1, only_save_img=True):
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
                             skiprows=range(1, skiprows),
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
            _index = index + first_batch
            self.save_data_to_binary_file(data, file_dir + 'images/image_data_batch_' + str(_index) + '.bin')
            if only_save_img:
                continue
            size = img_configs['max-size'][0] >> 1 # np.random.randint(1, 3)
            block_size = (size, size)
            block_dim = (img_configs['max-size'][0] // size, img_configs['max-size'][1] // size)
            data_batch = self.generate_data(data, block_dim, block_size, is_array=False)
            self.save_data_to_binary_file(data_batch, file_dir + 'data_batch_' + str(_index) + '.bin')
            print('Finish saving data batch ' + str(_index) + ' / num rows: ' + str(len(data_batch['data'])))
        return data
            
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def generate_data_from_image(self, image, block_dim, block_size, IMG_SIZE):
    
        new_dataset = {
            'data': [],
            'target': []
        }
        
        if psutil.cpu_percent() > 50:
            return new_dataset
        
        image = cv2.resize(image, IMG_SIZE, interpolation = cv2.INTER_AREA)
        # rotate image
        image = np.rot90(image, k=np.random.randint(0, 4))
        # cv2.imwrite('output/sample.png', image)
    
        blocks = self.split_image_to_blocks(image, block_dim, outlier_rate=img_configs['outlier-rate'])
        dropped_blocks, lost_block_labels, _ = self.random_drop_blocks(blocks)
        
        # recovered_image = self.merge_blocks(dropped_blocks)
        # cv2.imwrite('output/sample.png', recovered_image)
        lost_positions = set()
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                if lost_block_labels[i][j] == 1:
                    lost_positions.add((i, j))
        lost_positions = list(lost_positions)
        
        for x in range(block_dim[0] - 1):
            for y in range(block_dim[1] - 1):
                n_losts = np.sum(lost_block_labels[x:x+2,y:y+2])
                if n_losts <= 2:
                    if SystemRandom().uniform(0, 1) < 0.25:
                        subblocks = dropped_blocks[x:x+2,y:y+2]
                        recovered_image = self.merge_blocks(subblocks)
                        # cv2.imwrite('output/sample.png', recovered_image)
                        for i in range(0, 2):
                            for j in range(0, 2):
                                _x = x + i
                                _y = y + j
                                if lost_block_labels[_x, _y] == 0:
                                    index = np.zeros(4, dtype=np.int8)
                                    index[i * 2 + j] = 1
                                    new_dataset['data'].append([recovered_image, index])
                                    new_dataset['target'].append(1)
                                    if SystemRandom().uniform(0, 1) < 0.5:
                                        continue
                                    _subblocks = copy(subblocks)
                                    _subblocks[i][j] = np.rot90(copy(subblocks[i][j]), k=np.random.randint(1, 4))
                                    _recovered_image = self.merge_blocks(_subblocks)
                                    new_dataset['data'].append([_recovered_image, index])
                                    new_dataset['target'].append(0)
                                    # cv2.imwrite('output/sample.png', _recovered_image)
                                
                if n_losts >= 1 and n_losts <= 3 and len(lost_positions) > 1:
                    if SystemRandom().uniform(0, 1) < 0.35:
                        continue
                    subblocks = dropped_blocks[x:x+2,y:y+2]
                    # recovered_image = self.merge_blocks(subblocks)
                    # cv2.imwrite('output/sample.png', recovered_image)
                    for i in range(0, 2):
                        for j in range(0, 2):
                            _x = x + i
                            _y = y + j
                            if n_losts == 3:
                                if lost_block_labels[_x + (i + 1)% 2, _y + (j + 1)% 2] == 0:
                                    continue
                            if lost_block_labels[_x, _y] == 1:
                                index = np.zeros(4, dtype=np.int8)
                                index[i * 2 + j] = 1
                                rd_ids = lost_positions[SystemRandom().randint(0, len(lost_positions) - 1)]
                                while rd_ids[0] == _x and rd_ids[1] == _y:
                                    rd_ids = lost_positions[SystemRandom().randint(0, len(lost_positions) - 1)]
                                rd_block = np.rot90(blocks[rd_ids[0]][rd_ids[1]], 
                                                    k=SystemRandom().randint(0, 3))
                                cp_subblocks = copy(subblocks)
                                cp_subblocks[i][j] = rd_block
                                _recovered_image = self.merge_blocks(cp_subblocks)
                                new_dataset['data'].append([_recovered_image, index])
                                new_dataset['target'].append(0)
                                # cv2.imwrite('output/sample.png', _recovered_image)
        return new_dataset
    
    def generate_data(self, dataset, block_dim, block_size, n_jobs=1):
        HEIGHT = block_size[0] * block_dim[0]
        WIDTH = block_size[1] * block_dim[1]
        IMG_SIZE = (HEIGHT, WIDTH)
        
        new_dataset = {
            'data': [],
            'target': []
        }
        counts = [0, 0]
        
        params = []
        t = tqdm(dataset, desc='Generating data')
        for org_image in t:
            params.append((org_image, block_dim, block_size, IMG_SIZE))
            if len(params) == n_jobs:
                with Pool(n_jobs) as p:
                    results = p.starmap(self.generate_data_from_image, params)
                for result in results:
                    new_dataset['data'].extend(result['data'])
                    new_dataset['target'].extend(result['target'])
                    for r in result['target']:
                        counts[r] += 1
                params = []
                t.set_postfix(size="{}/{}".format(counts[0], counts[1]))
                    
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
    
    def convert_rgb_to_gray(self, rgb_image):
        """
        Convert rgb image to gray image
        :param rgb_image: rgb image
        :return: gray image
        """
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def split_image_to_blocks(self, image, block_dim=(2, 2), smoothing=False, outlier_rate=0):
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
                if smoothing:
                    block = cv2.GaussianBlur(block, (3, 3), 0)
                if outlier_rate > 0:
                    for _i in range(block_size[0]):
                        for _j in range(block_size[1]):
                            if np.random.uniform() < outlier_rate:
                                block[_i, _j][np.random.randint(0, 3)] = np.random.randint(0, 255)
                                
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
        _indices = []
        for i in range(n_rows):
            for j in range(n_cols):
                _indices.append((i, j))
        pos = [i for i in range(1, len(_indices))]
        np.random.shuffle(pos)
        pos = [0] + pos
        indices = [_indices[i] for i in pos]
        count = 0
        
        for i in range(n_rows):
            for j in range(n_cols):
                x, y = indices[count]
                block = blocks[x][y]
                if rotate and i != 0 and j != 0:
                    block = np.rot90(block, k=np.random.randint(4))
                shuffled_blocks[i][j] = block
                shuffled_labels[x][y] = i * n_cols + j
                count += 1
        return shuffled_blocks, shuffled_labels
    
    def random_drop_blocks(self, blocks, prob=None):
        """
        Random drop blocks
        :param blocks: blocks
        :param prob: probability of drop
        :return: dropped blocks
        """
        if prob is None:
            prob = SystemRandom().uniform(0, 1) * 0.4 + 0.3
        n_rows, n_cols = blocks.shape[0], blocks.shape[1]
        n_steps = prob * n_rows * n_cols
        masked = np.zeros((n_rows, n_cols), dtype=np.uint8)
        dx = [0, 1, 0, -1]
        dy = [-1, 0, 1, 0]
        
        block_size = blocks[0][0].shape
        dropped_blocks = np.empty(blocks.shape, dtype=np.ndarray)
        lost_block_labels = np.zeros((n_rows, n_cols))
        next_position = set()
        x, y = SystemRandom().randint(0, n_rows - 1), SystemRandom().randint(0, n_cols - 1)
        masked[x][y] = 1
        for i in range(4):
            _x, _y = x + dx[i], y + dy[i]
            if 0 <= _x < n_rows and 0 <= _y < n_cols:
                next_position.add((_x, _y))
        
        for i in range(int(n_steps)):
            next_pos = list(next_position)[SystemRandom().randint(0, len(next_position) - 1)]
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