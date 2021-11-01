import h5py

def load_usps(path):
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_train = train.get('data')[:]
        y_train = train.get('target')[:]
        test = hf.get('test')
        X_test = test.get('data')[:]
        y_test = test.get('target')[:]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, y_train, X_test, y_test
