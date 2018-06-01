import numpy as np
from sklearn.metrics import accuracy_score
import json

import data_loader
import decision_tree

# load data
X_train, X_test, y_train, y_test = data_loader.discrete_2D_iris_dataset()

# set classifier
dTree = decision_tree.DecisionTree()

# training
dTree.train(X_train, y_train)
y_est_train = dTree.predict(X_train)
train_accu = accuracy_score(y_est_train, y_train)
print('train_accu', train_accu)

# testing
y_est_test = dTree.predict(X_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu)



# print
dTree.print_tree()

# save
json.dump({'train_accu': train_accu, 'test_accu': test_accu},
			open('decision_tree.json', 'w'))