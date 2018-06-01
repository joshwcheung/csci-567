import numpy as np

import data_loader
import decision_tree

###############
# Toy example #
###############
'''
Toy example

dim_1
 ┃
 ╋       ○
 ┃
 ╋   ×       ○
 ┃
 ╋       ×
 ┃
━╋━━━╋━━━╋━━━╋━ dim_0

Print the tree and check the result by yourself!
             
'''
# data
features, labels = data_loader.toy_data_3()

# build the tree
dTree = decision_tree.DecisionTree()
dTree.train(features, labels)

# print
dTree.print_tree()