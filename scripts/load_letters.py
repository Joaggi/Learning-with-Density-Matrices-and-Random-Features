import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split


def load_letters(path):
    letters = pd.read_csv(path, header=None)
    #letters = pd.read_csv("letter-recognition.data", header=None)
    print(letters.iloc[:,1:].head())
    print(letters.iloc[:,1:].describe())
    #print(letters.iloc[:,1:].info())
    print(np.unique(letters.iloc[:,0]))

    vector = letters.values[:,1:]
    labels = letters.values[:,0]

    for index, i in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']):
        labels[labels==i] = index
        #print(index, " ",  i)

    labels=  np.array(labels, dtype=np.int32)


    X_train, X_test, y_train, y_test = train_test_split(vector, labels, test_size=0.3, random_state=42, stratify=labels)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, y_train, X_test, y_test
