import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split


def load_forest(path):
    dataset = pd.read_csv(path, nrows=100, compression='gzip',error_bad_lines=False)

    dataset = dataset.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
            dataset[:,:-1], dataset[:, -1], test_size=0.33, random_state=42)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print("shape X_train : ", X_train.shape)
    print("shape y_train : ", y_train.shape)
    print("shape X_test : ", X_test.shape)
    print("shape y_test : ", y_test.shape)


    y_train[y_train == 1] = 0
    y_test[y_test == 1] = 0

    y_train[y_train == 2] = 1
    y_test[y_test == 2] = 1

    y_train[y_train == 5] = 2
    y_test[y_test == 5] = 2

   
    return X_train, y_train, X_test, y_test
