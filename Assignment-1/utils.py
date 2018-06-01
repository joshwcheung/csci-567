from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    if 2 * tp + fp + fn == 0:
        return 0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return f1


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    x = np.array(features)
    x_prime = np.array(features)
    for i in range(2, k + 1):
        x_prime = np.hstack((x_prime,  x ** i))
    return x_prime.tolist()


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x - y) ** 2 for x, y in zip(point1, point2)]
    distance = np.sqrt(sum(distance))
    return distance


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x * y) for x, y in zip(point1, point2)]
    distance = sum(distance)
    return distance


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    distance = [(x - y) ** 2 for x, y in zip(point1, point2)]
    distance = sum(distance)
    distance = -np.exp(-0.5 * distance)
    return distance


class NormalizationScaler:
    def __init__(self):
        pass
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized = []
        for sample in features:
            if all(x == 0 for x in sample):
                normalized.append(sample)
            else:
                denom = float(np.sqrt(inner_product_distance(sample, sample)))
                sample_normalized = [x / denom for x in sample]
                normalized.append(sample_normalized)
        return normalized


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min, self.max = None, None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        feat_array = np.array(features)
        if self.min is None or self.max is None:
            self.min = np.amin(feat_array, axis=0)
            self.max = np.amax(feat_array, axis=0)
        normalized = (feat_array - self.min) / (self.max - self.min)
        return normalized.tolist()

