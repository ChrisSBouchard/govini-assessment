import argparse
from src.data import create_dataset_subset
from src.aiml import train, test

def main():
    parser = argparse.ArgumentParser(description='create train/test datasets or train, tune, test, or run binary classification models')
    parser.add_argument('action', choices=['dataset', 'train', 'test'], 
                        help='the action to run on the given dataset one of dataset, train, test')
    args = parser.parse_args()

    if args.action == 'dataset':
        create_dataset_subset()
    if args.action == 'train':
        train()
    if args.action == 'test':
        test()

if __name__ == "__main__":
    main()