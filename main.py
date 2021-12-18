
import numpy as np
from src.data_helper import DataProcessor
from configs import configs, img_configs
from src.trainer import Trainer
from models.ProNet import ProNet
from models.ProNet2 import ProNet2
import cv2

def get_dataset(file_dir, file_name, iter, saved=False):
    if configs['preprocess']:
        dataset = DataProcessor.load_data_from_binary_file(file_dir, file_name)[:200]
        dataset = DataProcessor.generate_data(dataset, 
                                        (4,4),
                                        img_configs['block-size'],
                                        n_jobs=8)
        
        DataProcessor.save_data_to_binary_file(dataset, "input/data/dataset_{}.bin".format(iter))   
        trainset, testset = DataProcessor.split_dataset(dataset, 0.98, saved=False)
    else:
        dataset = DataProcessor.load_data_from_binary_file(file_dir, file_name)
        trainset, testset = DataProcessor.split_dataset(dataset, 0.95, saved=False)
    return trainset, testset    

def main():
    configs['preprocess'] = True
    configs['num-dataset'] = 2
    file_dir = "input/data/images_data/"
    trainer = Trainer(model=ProNet2(img_configs['image-size']), lr=0.0001, loss='bce', optimizer='adas', batch_size=64, epochs=3)
    # trainer.model.load_checkpoint(1, 616)
    for i in range(configs['num-dataset']):
        file_name = "image_data_batch_{}.bin".format(i)
        trainset, testset = get_dataset(file_dir, file_name, i, saved=False)
        print("Train set size: ", len(trainset['data']))
        print("Test set size: ", len(testset['data']))
        ''' count the number of each class in the dataset '''
        target_count = [0, 0, 0]
        for i in range(len(trainset['target'])):
            if trainset['target'][i] == (1, 1):
                target_count[0] += 1
            elif trainset['target'][i] == (1, 0):
                target_count[1] += 1
            elif trainset['target'][i] == (0, 0):
                target_count[2] += 1
        print('class distribution: {}/{}/{}'.format(target_count[0], 
                                                    target_count[1],
                                                    target_count[2]))
        ''' print sample '''
        _id = np.random.randint(0, len(trainset['data']))
        sample_img = trainset['data'][_id][0]
        print('target: {} / Angle: {}'.format(trainset['target'][_id][0], trainset['target'][_id][1]))
        cv2.imwrite("output/sample.png", sample_img)
        trainer.train_loader = trainset
        trainer.test_loader = testset
        trainer.train()
    
if __name__ == "__main__":
    main()