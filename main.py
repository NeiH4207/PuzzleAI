import _pickle as cPickle
from numpy.core.shape_base import block
from src.data_helper import DataHelper
from configs import configs, img_configs
import cv2
import numpy as np
from src.trainer import Trainer
from models.vgg import VGG
from models.ProNet import ProNet
import torch.nn as nn

def main():
    DataProcessor = DataHelper()
    if configs['preprocess']:
        file_dir = "/sharedfiles/cifar-10-batches-py/data_batch_1"
        dataset = DataProcessor.load_data_from_binry_file(file_dir)[b'data']
        dataset = DataProcessor.transform(dataset, 
                                        img_configs['block-dim'],
                                        img_configs['block-size'])
        
        DataProcessor.save_data_to_binary_file(dataset, "input/dataset.bin")     
        trainset, testset = DataProcessor.split_dataset(dataset, 0.9, saved=True)
    else:
        trainset, testset = DataProcessor.load_train_test_dataset(file_dir="input/")
        
    print("Train set size: ", len(trainset['data']))
    print("Test set size: ", len(testset['data']))
    # img_sample, label = trainset['data'][0][0], trainset['target'][0]
    # print(img_sample)
    # print(trainset['data'][0][1])
    # print(label)
    # cv2.imwrite("output/img_sample.png", img_sample)
    # testset = {
    #     'data':testset['data'][:1000],
    #     'target':testset['target'][:1000]
    # }
    trainer = Trainer(model=ProNet(1, img_configs['image-size']), loss=nn.MSELoss, optimizer='adam',
                      train_loader=trainset, test_loader=testset, batch_size=64, epochs=10)
    trainer.load_model(1, 0)
    # trainer.train()
    trainer.test()
    
if __name__ == "__main__":
    main()