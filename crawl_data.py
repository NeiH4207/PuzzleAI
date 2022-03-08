
import argparse
from src.data_helper import DataProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='batch size')
    parser.add_argument('-s', '--skiprows', type=int, default=26560, help='skip size')
    parser.add_argument('-d', '--data-dir', type=str, default='data', help='data directory')
    parser.add_argument('-o', '--output-dir', type=str, default='input/data/64x64/', help='output directory')
    parser.add_argument('-t', '--type', type=str, default='train', help='data type')
    parser.add_argument('-f', '--first-batch', type=int, default=0, help='first batch')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    file_dir = "input/2017_11/" + args.type + "/"
    # file_dir = "input/data/stl10_binary/"
    file_name = "images.csv"
    DataProcessor.read_url_csv(file_dir, file_name, 
                               chunksize=args.batch_size, 
                               skiprows=args.skiprows,
                               first_batch=args.first_batch, 
                               output_dir=args.output_dir,
                               only_save_img=True)
    
if __name__ == "__main__":
    main()
