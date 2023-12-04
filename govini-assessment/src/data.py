import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset_subset():
    '''
    Randomly splits the dataset and saves a train dataset and test dataset
    '''
    # Path to base parquet dataset
    path = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/base/ml_dataset.parquet"
    df = pd.read_parquet(path, engine='fastparquet')
    proportions = df['labels'].value_counts()
    proportion = proportions[1] / (proportions[0] + proportions[1])
    print(f"Total Class Proportion: {proportion}")
    proportion = 0
    while proportion < 0.02:
        train, test = train_test_split(df, test_size=0.5)
        proportions = train['labels'].value_counts()
        proportion = proportions[1] / (proportions[0] + proportions[1])
        print(f"Train Class Proportion: {proportion}")
    
    split_path = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/split/"

    train.to_parquet(f"{split_path}train.parquet", engine='fastparquet')
    test.to_parquet(f"{split_path}test.parquet", engine='fastparquet')
