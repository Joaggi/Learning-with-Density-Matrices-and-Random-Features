import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split

from keras.datasets import mnist

def load_mnist(path):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((60000,784))
    X_test = X_test.reshape((10000,784))   
    return X_train, y_train, X_test, y_test
