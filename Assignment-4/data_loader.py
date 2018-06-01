from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


def toy_dataset(cluster_size=2, sample_per_cluster=50):
    # TODO: add sample size ease
    np.random.seed(42)
    N = cluster_size*sample_per_cluster
    y = np.zeros(N)
    x = np.random.standard_normal(size=(N, 2))
    for i in range(cluster_size):
        theta = 2*np.pi*i/cluster_size
        x[i*sample_per_cluster:(i+1)*sample_per_cluster] = x[i*sample_per_cluster:(i+1)*sample_per_cluster] + \
            (cluster_size*np.cos(theta), cluster_size*np.sin(theta))
        y[i*sample_per_cluster:(i+1)*sample_per_cluster] = i
    return x, y


def load_digits():
    digits = datasets.load_digits()
    x = digits.data/16
    x = x.reshape([x.shape[0], -1])
    y = digits.target
    return train_test_split(x, y, random_state=42, test_size=0.25)
