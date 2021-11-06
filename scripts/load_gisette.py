import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def load_gisette(path, random_state = 0):


    num = 6000
    
    train_data = pd.read_csv(f"{path}gisette_train.data", header=None, sep=" ").iloc[0:num,:5000]
    train_labels = pd.read_csv(f"{path}gisette_train.labels", header=None, sep=" ").iloc[0:num]
    test_data = pd.read_csv(f"{path}gisette_valid.data", header=None, sep=" ").iloc[0:num,:5000]
    test_labels = pd.read_csv(f"{path}gisette_valid.labels", header=None, sep=" ").iloc[0:num]


    print(train_data.head())
    print(train_data.describe())
    print(train_data.info())
    print(np.unique(train_labels))
    print(train_data.shape)

    train_labels[train_labels == -1] = 0
    test_labels[test_labels == -1] = 0
 
    X_train = train_data.values
    X_test = test_data.values
    y_train = train_labels.values
    y_test = test_labels.values

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    pca = PCA(n_components=400, random_state = random_state)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test
