from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split

import json
import numpy as np


def iris_dataset():
    iris = load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=0.8, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def toy_data_binary():
    data = make_classification(n_samples=500, 
                               n_features=2, 
                               n_informative=1, 
                               n_redundant=0, 
                               n_repeated=0, 
                               n_classes=2, 
                               n_clusters_per_class=1, 
                               class_sep=1., 
                               random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], 
                                                        train_size=0.7, 
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def toy_data_multiclass_3_classes_():
    data = make_blobs(n_samples=500, 
                      n_features=2,
                      random_state=42, 
                      centers=[[0, 1], [2, 0], [0, -2]])
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], 
                                                        train_size=0.7, 
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def toy_data_multiclass_3_classes_separable():
    data = make_blobs(n_samples=500, 
                      n_features=2,
                      random_state=42, 
                      centers=[[0, 5], [10, 0], [0, -5]], 
                      cluster_std=0.5)
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], 
                                                        train_size=0.7, 
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def toy_data_multiclass_3_classes_non_separable():
    data = make_blobs(n_samples=500, 
                      n_features=2,
                      random_state=42, 
                      centers=[[0, 5], [10, 0], [0, -5]], 
                      cluster_std=3.5)
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], 
                                                        train_size=0.7, 
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def toy_data_multiclass_5_classes():
    data = make_blobs(n_samples=500, 
                      n_features=2,
                      random_state=42, 
                      centers=[[0, 5], [10, 0], [0, -5], [-5, 2], [4, 1]], 
                      cluster_std=2.)
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], 
                                                        train_size=0.7, 
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def data_loader_mnist(dataset='mnist_subset.json'):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    return np.asarray(train_set[0]), \
           np.asarray(test_set[0]), \
           np.asarray(train_set[1]), \
           np.asarray(test_set[1])
    # Xtrain = train_set[0]
    # Ytrain = train_set[1]
    # Xvalid = valid_set[0]
    # Yvalid = valid_set[1]
    # Xtest = test_set[0]
    # Ytest = test_set[1]

    # return np.array(Xtrain).reshape(-1, 1, 28, 28), np.array(Ytrain), np.array(Xvalid).reshape(-1, 1, 28, 28),\
    #        np.array(Yvalid), np.array(Xtest).reshape(-1, 1, 28, 28), np.array(Ytest)
