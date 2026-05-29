import numpy as np
import urllib.request
import gzip
import os

def download_mnist(path='data'):
    os.makedirs(path, exist_ok=True)
    
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz'
    }
    
    for name, file in files.items():
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f'Downloading {file}...')
            urllib.request.urlretrieve(base_url + file, filepath)

def load_mnist(path='data'):
    download_mnist(path)
    
    def parse_images(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(16)                              # skip header
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 784).astype(np.float32) / 255.0     # normalize values
    
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(8)                               # skip header
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    X_train = parse_images('data/train-images-idx3-ubyte.gz')
    y_train = parse_labels('data/train-labels-idx1-ubyte.gz')
    X_test  = parse_images('data/t10k-images-idx3-ubyte.gz')
    y_test  = parse_labels('data/t10k-labels-idx1-ubyte.gz')
    
    return X_train, y_train, X_test, y_test

def get_batches(X, y, batch_size=32):
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]      # doesnt generate all batches at once to store in memory; one at a time and waits for next generation of batch