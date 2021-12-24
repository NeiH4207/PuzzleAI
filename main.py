

import numpy as np
from src.data_helper import DataProcessor
from configs import configs, img_configs
from src.trainer import Trainer
# from models.ProNet import ProNet
from models.SimpleProNet import ProNet
from models.ProNet2 import ProNet2
import cv2
from utils import *

def get_dataset(file_dir, file_name, iter, saved=False):
    if configs['preprocess']:
        dataset = None
        dataset = DataProcessor.load_data_from_binary_file(file_dir, file_name)[:30]
        dataset = DataProcessor.generate_data(dataset, 
                                        (4,4),
                                        img_configs['block-size'],
                                        n_jobs=8)
        
        DataProcessor.save_data_to_binary_file(dataset, "input/data/dataset_{}.bin".format(iter))
        trainset, testset = DataProcessor.split_dataset(dataset, 0.98, saved=False)
    else:
        dataset = DataProcessor.load_data_from_binary_file("input/data/","dataset_{}.bin".format(iter))[:1000]
        trainset, testset = DataProcessor.split_dataset(dataset, 0.97, saved=False)
    return trainset, testset    

def main():
    configs['preprocess'] = False or True
    configs['num-dataset'] = 20
    file_dir = "input/data/images_data/"
    trainer = Trainer(model=ProNet(img_configs['image-size']), 
                      lr=0.0001, 
                      loss='bce', 
                      optimizer='adas', 
                      batch_size=64, 
                      n_repeats=2)
    trainer.model.load_checkpoint(0, 3500)
    for i in range(1, configs['num-dataset']):
        file_name = "image_data_batch_{}.bin".format(i)
        trainset, testset = get_dataset(file_dir, file_name, i, saved=False)
        print("Train set size: ", len(trainset['data']))
        print("Test set size: ", len(testset['data']))
        ''' print sample '''
        _id = np.random.randint(0, len(trainset['data']))
        sample_img = trainset['data'][_id][0]
        print('target: {}'.format(trainset['target'][_id]))
        cv2.imwrite("output/sample.png", sample_img)
        trainer.train_loader = trainset
        trainer.test_loader = testset
        trainer.train()
        trainer.train_loader = None
        trainer.test_loader = None
        trainset, testset = None, None
    
    
if __name__ == "__main__":
    main()
