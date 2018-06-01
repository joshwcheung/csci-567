from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        n = len(features)
        x = numpy.hstack((numpy.ones((n, 1)), numpy.array(features)))
        y = numpy.array(values)
        self.weights = numpy.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        
    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        n = len(features)
        x = numpy.hstack((numpy.ones((n, 1)), numpy.array(features)))
        y = x.dot(self.weights)
        return y

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.tolist()


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        n = len(features)
        m = len(features[0]) + 1
        x = numpy.hstack((numpy.ones((n, 1)), numpy.array(features)))
        y = numpy.array(values)
        diag = self.alpha * numpy.identity(m)
        self.weights = numpy.linalg.inv(x.T.dot(x) + diag).dot(x.T).dot(y)

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        n = len(features)
        x = numpy.hstack((numpy.ones((n, 1)), numpy.array(features)))
        y = x.dot(self.weights)
        return y

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.tolist()


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
