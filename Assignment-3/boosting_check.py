import numpy as np
from sklearn.metrics import accuracy_score

import data_loader
import decision_stump
import boosting



#################
# Toy example 1 #
#################
'''
Toy example of XOR

          ┃
    ○     ╋     ×
          ┃
          ┃
━━━━╋━━━━━╋━━━━━╋━━━
          ┃
          ┃
    ×     ╋     ○
          ┃

Given a simple set of decision stumps:
s in {1, -1}
b in {-2, -0.5, 0.5, 2}
d in {0, 1}

'''
# data
features_1, labels_1 = data_loader.toy_data_1()
# clfs
h_set = set()
s_set = {1, -1}
b_set = {-2, -0.5, 0.5, 2}
d_set = {0, 1}
for s in s_set:
	for b in b_set:
		for d in d_set:
			h_set.add(decision_stump.DecisionStump(s,b,d))
# boost
Ada_1 = boosting.AdaBoost(h_set, T=1)
Ada_1.train(features_1, labels_1)
Log_1 = boosting.LogitBoost(h_set, T=1)
Log_1.train(features_1, labels_1)

# check
print('━━━━━━━━━━ Toy example 1 ━━━━━━━━━━')
print('This toy example checks the format. Any of the stump is correct.')
print('(Can you explain why?)')
print('Ada_1: s = {:01d}, b = {:.1f}, d = {:01d}'.format(
	Ada_1.clfs_picked[0].s, Ada_1.clfs_picked[0].b, Ada_1.clfs_picked[0].d))
print('Log_1: s = {:01d}, b = {:.1f}, d = {:01d}'.format(
	Log_1.clfs_picked[0].s, Log_1.clfs_picked[0].b, Log_1.clfs_picked[0].d))
if Ada_1.betas[0] == 0 and Log_1.betas[0] == 0.5:
	print('Betas are correct')
else:
	print('▁▂▃▄▅▆▇█ Betas are not correct █▇▆▅▄▃▂▁')


#################
# Toy example 2 #
#################
'''
Toy example of another XOR (linearly transformed from toy example 1)
		  
          ┃
          ×     
          ┃
          ┃
━━━━○━━━━━╋━━━━━○━━━
          ┃
          ┃
          ×     
          ┃

Given a simple set of decision stumps:
s in {1, -1}
b in {-2, -0.5, 0.5, 2}
d in {0, 1}

'''
# data
features_2, labels_2 = data_loader.toy_data_2()
# clfs
h_set = set()
s_set = {1, -1}
b_set = {-2, -0.5, 0.5, 2}
d_set = {0, 1}
for s in s_set:
	for b in b_set:
		for d in d_set:
			h_set.add(decision_stump.DecisionStump(s,b,d))
# boost
Ada_2_2 = boosting.AdaBoost(h_set, T=2)
Ada_2_2.train(features_2, labels_2)
Ada_2_3 = boosting.AdaBoost(h_set, T=3)
Ada_2_3.train(features_2, labels_2)
Log_2_3 = boosting.LogitBoost(h_set, T=3)
Log_2_3.train(features_2, labels_2)

# check
print('━━━━━━━━━━ Toy example 2 ━━━━━━━━━━')
Ada_2_2_acc = accuracy_score(Ada_2_2.predict(features_2), labels_2)
Ada_2_3_acc = accuracy_score(Ada_2_3.predict(features_2), labels_2)
Log_2_3_acc = accuracy_score(Log_2_3.predict(features_2), labels_2)
print('Ada:', Ada_2_2_acc, Ada_2_3_acc)
print('Log:', Log_2_3_acc)
if Ada_2_2_acc == 0.75 and Ada_2_3_acc == 1 and Log_2_3_acc == 1:
	print('Correct training accuracies')
else:
	print('▁▂▃▄▅▆▇█ Incorrect training accuracies █▇▆▅▄▃▂▁')