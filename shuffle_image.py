import cv2
from src.recover.environment import GameInfo
from src.data_helper import DataHelper

import argparse

from utils import tuple_type

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./input/original_images/')
    parser.add_argument('--output-path', type=str, default='./input/shuffle_images')
    parser.add_argument('-f', '--filename', type=str, default='natural_1.jpg')
    parser.add_argument('--file-name-out', type=str, default=None)
    parser.add_argument('-s', '--block-size', type=int, default=64)
    parser.add_argument('-d', '--block-dim', type=tuple_type, default=(3, 4))
    parser.add_argument('--mode', type=str, default='rgb')
        
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    DataProcess = DataHelper()
    block_size = tuple((args.block_size, args.block_size))
    block_dim = tuple(args.block_dim)
    image_size = block_size[1] * block_dim[1], block_size[0] * block_dim[0]
    image = cv2.imread(args.image_path + args.filename)
        
    if args.mode == 'bw':
        image = DataProcess.convert_rgb_to_gray(image)
        
    image = cv2.resize(image, image_size)
    blocks = DataProcess.split_image_to_blocks(image, block_dim)
    blocks, labels = DataProcess.shuffle_blocks(blocks, rotate=True)
    image = DataProcess.merge_blocks(blocks)
    
    if args.file_name_out is None:
        args.file_name_out = args.filename.split('.')[0] + '.png'
    cv2.imwrite(args.output_path + '/' + args.file_name_out, image)
    print('image saved in {}'.format(args.output_path + '/' + args.file_name_out))
    
    game_info = GameInfo()
    game_info.name = args.filename.split('.')[0]
    game_info.block_dim = block_dim
    game_info.max_n_selects = block_dim[0] * block_dim[1]
    game_info.select_swap_ratio = [5, 5]
    game_info.image_size = image_size
    game_info.max_image_point_value = [255, 255, 255]
    game_info.original_block_size = block_size
    game_info.original_blocks = blocks.tolist()
    game_info.mode = 'rgb'
    game_info.save_to_json()
    
if __name__ == "__main__":
    main()