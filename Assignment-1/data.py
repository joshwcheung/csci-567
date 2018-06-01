import math
from typing import List, Tuple
import numpy

''' All data utilities are here'''


def generate_data_part_1() -> Tuple[List[List[float]], List[float]]:
    features = [[i / 10.] for i in range(0, 10)]
    values = [2 * (x[0] ** 1.2) + 1 for x in features]
    return features, values


def generate_data_part_2() -> Tuple[List[List[float]], List[float]]:
    features = [[i / 10.0] for i in range(-10, 20)]
    values = [math.exp(x[0]) for x in features]
    return features, values


def generate_data_part_3() -> Tuple[List[List[float]], List[float]]:
    from sklearn.datasets import load_iris
    features, labels = load_iris(return_X_y=True)
    features = features.astype(float).tolist()
    labels = labels.astype(int).tolist()
    return features, labels


def generate_data_cancer() -> Tuple[List[List[float]], List[int]]:
    from sklearn.datasets import load_breast_cancer
    features, labels = load_breast_cancer(return_X_y=True)
    features = features.astype(float).tolist()
    labels = labels.astype(int).tolist()
    return features, labels


def generate_data_perceptron(nb_features=2, seperation=2) -> Tuple[List[List[float]], List[int]]:
    ''' Generates data for perceptron problem'''
    ''' 
        nb_features = 2, seperation=2 generates linearly seperable data
        nb_features = 2, seperation=1 generates linearly seperable data
        
        Also note that it appends 1 to feature vector to account for bias term

    '''
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=100, n_features=nb_features, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=2,
                               weights=None, flip_y=0.00, class_sep=seperation, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=500)
    y = 2 * y - 1
    x = numpy.insert(x, 0, 1, axis=1)

    features = x.astype(float).tolist()
    labels = y.astype(int).tolist()
    return features, labels
