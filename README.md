# Classification of Google Patents

## Description
This project provides the means of training, tuning, and testing binary classification models using the publicly available Google patents dataset.

## Usage
Create a directory named *datasets* with *base* and *split* subdirectories.  Place the dataset parquet file in the *base* directory
There are three actions currently available: dataset, train, and test

You can run an action by running `python main.py [action]`.

Ensure you have installed the dependencies from `requirements.txt`

### dataset
This action creates a datasplit ensuring the positive class label is above 2% of the train dataset.  The train dataset will be used for training, and any models created will never see the test set during training.

### train
This action trains model using the entirety of the training set.

Currently, this trains a logistic regression model as well as creates a `CountVectorizer` for converting text to a matrix of tokens.  Both the model and vectorizer are saved.

### test
Loads the model and vectorizer and predicts output from the never-seen test dataset then prints a classification report.

## TODO
 - Implement a means of tuning models
 - Experiment with different model types
 - Provide a means of running predictions on new patent data