from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split

import numpy as np

def toy_data_1():
	features = [[1.,1.], [-1., 1.], [-1.,-1.], [1.,-1.]]
	labels = [1, -1,  1, -1]
	return features, labels

def toy_data_2():
	features = [[0., 1.414], [-1.414, 0.], [0., -1.414], [1.414, 0.]]
	labels = [1, -1, 1, -1]
	return features, labels

def toy_data_3():
	features = [[1,2], [2,1], [2,3], [3,2]]
	labels = [0, 0, 1, 1]
	return features, labels


def binary_iris_dataset():
	iris = load_iris()
	X = iris.data[50: , ]
	y = iris.target[50: , ]
	y = y * 2 - 3

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
	return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

def discrete_2D_iris_dataset():
	iris = load_iris()
	X = iris.data[:, [0,1]]
	y = iris.target
	
	X_discrete = np.ones(X.shape)
	X_discrete[X[:,0]<5.45, 0] = 0
	X_discrete[X[:,0]>=6.15, 0] = 2
	X_discrete[X[:,1]<2.8, 1] = 0
	X_discrete[X[:,1]>=3.45, 1] = 2

	X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, train_size=0.8, random_state=3)
	return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()