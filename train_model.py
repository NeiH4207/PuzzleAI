

import numpy as np
from src.data_helper import DataProcessor
from configs import configs, img_configs
from src.trainer import Trainer
# from models.ProNet import ProNet
from models.SimpleProNet import ProNet
from models.ProNet2 import ProNet2
from models.VGG import VGG
from models.dla import DLA
from models.densenet import densenet_cifar

import cv2
from utils import *

def get_dataset(file_dir, file_name, iter, saved=False):
    if configs['preprocess']:
        dataset = None
        dataset = DataProcessor.load_data_from_binary_file(file_dir, file_name)
        dataset = DataProcessor.generate_data(dataset, 
                                        img_configs['block-dim'],
                                        img_configs['block-size'],
                                        n_jobs=8)
        
        # DataProcessor.save_data_to_binary_file(dataset, "input/data/64x64/train_batchs/dataset_{}.bin".format(iter))
        trainset, testset = DataProcessor.split_dataset(dataset, 0.95, saved=False)
    else:
        dataset = DataProcessor.load_data_from_binary_file("input/data/64x64/train_batchs/","dataset_{}.bin".format(iter))
        trainset, testset = DataProcessor.split_dataset(dataset, 0.97, saved=False)
    return trainset, testset

def main():
    configs['preprocess'] = True
    configs['num-dataset'] = 200
    file_dir = "input/data/64x64/images/"
    trainer = Trainer(model=VGG('VGG7'), 
                      lr=0.003, 
                      loss='bce', 
                      optimizer='adas', 
                      batch_size=256, 
                      n_repeats=1,
                      save_every=500
                      )
    
    # trainer.model.load(0, 1580)

    for i in range(0, configs['num-dataset']):
        file_name = "image_data_batch_{}.bin".format(i)
        trainset, testset = get_dataset(file_dir, file_name, i, saved=False)
        print("Train set size: ", len(trainset['data']))
        print("Test set size: ", len(testset['data']))
        trainer.train_loader = trainset
        trainer.test_loader = testset
        trainer.train()
        trainer.train_loader = None
        trainer.test_loader = None
        trainset, testset = None, None
    
if __name__ == "__main__":
    main()
