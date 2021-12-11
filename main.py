
from src.data_helper import DataHelper
from configs import configs, img_configs
from src.trainer import Trainer
from models.ProNet import ProNet
import cv2

def main():
    configs['preprocess'] = True
    DataProcessor = DataHelper()
    file_dir = "input/data/images_2017_11/2017_11/test/"
    file_dir = "input/data/images_data/"
    # file_dir = "/sharedfiles/cifar-10-batches-py/"
    file_name = "image_data_batch_0.bin"
    if configs['preprocess']:
        dataset = DataProcessor.load_data_from_binary_file(file_dir, file_name)
        dataset = DataProcessor.generate_data(dataset, 
                                        img_configs['block-dim'],
                                        img_configs['block-size'], is_array=False)
        
        DataProcessor.save_data_to_binary_file(dataset, "input/dataset2x2.bin")     
        trainset, testset = DataProcessor.split_dataset(dataset, 0.98, saved=True)
    else:
        dataset = DataProcessor.load_data_from_binary_file(file_dir, file_name)
        trainset, testset = DataProcessor.split_dataset(dataset, 0.95, saved=False)
        # trainset, testset = DataProcessor.load_train_test_dataset(file_dir="input/")
        
    print("Train set size: ", len(trainset['data']))
    print("Test set size: ", len(testset['data']))
    ''' count the number of each class in the dataset '''
    target_count = [0, 0]
    for i in range(len(trainset['target'])):
        target_count[trainset['target'][i][0]] += 1
    print('class distribution: {}/{}'.format(target_count[0], target_count[1]))
    ''' print sample '''
    print("Image shape: ", trainset['image_size'][0])
    _id = 13
    sample_img = trainset['data'][_id][0]
    print('target: {} / Angle: {}'.format(trainset['target'][_id][0], trainset['target'][_id][1]))
    cv2.imwrite("output/sample.png", sample_img)
    trainer = Trainer(model=ProNet(trainset['image_size'][0]), lr=0.0001, loss='bce', optimizer='adas',
                      train_loader=trainset, test_loader=testset, batch_size=64, epochs=10)
    trainer.model.load_checkpoint(1, 616)
    trainer.train()
    
if __name__ == "__main__":
    main()