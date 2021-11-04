import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split

from load_cifar_10 import cifar10

def load_cifar(path):

    train_images, train_labels, test_images, test_labels = cifar10(path='data', is_one_hot=False)

    print(train_images.shape)
    print(np.unique(train_labels))


    train_images.shape


    random_train = np.random.choice(range(train_images.shape[0]), 50000, replace=False)
    random_test = np.random.choice(range(test_images.shape[0]), 10000, replace=False)


    X_train = train_images[random_train,:]
    y_train = train_labels[random_train]
    X_test = test_images[random_test,:]
    y_test = test_labels[random_test]


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print("shape X_train : ", X_train.shape)
    print("shape y_train : ", y_train.shape)
    print("shape X_test : ", X_test.shape)
    print("shape y_test : ", y_test.shape)


    return X_train, y_train, X_test, y_test


