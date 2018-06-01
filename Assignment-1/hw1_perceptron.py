from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        n = len(features)
        x = np.array(features)
        converged = False
        step = 0
        while not converged and step < self.max_iteration:
            w = np.array(self.w)
            y = x.dot(w.T)
            w_norm = np.sqrt(sum([wj ** 2 for wj in w]))
            g = self.margin / 2
            eps = 1e-5
            
            y /= (w_norm + eps)
            y = [1 if yi >= g else -1 if yi <= g else 0 for yi in y]
            updated = False
            for i in range(n):
                if y[i] != labels[i]:
                    x_norm = np.sqrt(sum([xj ** 2 for xj in x[i]]))
                    self.w = self.w + labels[i] * x[i] / (x_norm + eps)
                    updated = True
            converged = not updated
            step += 1
        return converged
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        
        x = np.array(features)
        w = np.array(self.w)
        y = x.dot(w.T) / np.sqrt(sum([wj ** 2 for wj in w]))
        y = [1 if yi >= 0 else -1 for yi in y]
        return y

    def get_weights(self) -> List[float]:
        return self.w
    
