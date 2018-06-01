from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        labels = []
        for test in features:
            distances = []
            for train in self.features:
                distances.append(self.distance_function(train, test))
            indexes = numpy.argpartition(distances, self.k)
            votes = {}
            if self.k < 1:
                self.k = 1
            for i in range(self.k):
                label = self.labels[indexes[i]]
                if label not in votes.keys():
                    votes[label] = 1
                else:
                    votes[label] += 1
            labels.append(max(votes, key=votes.get))
        return labels


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
