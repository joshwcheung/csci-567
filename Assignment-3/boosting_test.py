import numpy as np
from sklearn.metrics import accuracy_score
import json

import data_loader
import decision_stump
import boosting

# load data
X_train, X_test, y_train, y_test = data_loader.binary_iris_dataset()
assert(len(X_train) == 80)
assert(len(X_train[0]) == 4)
assert(len(X_test) == 20)
assert(len(X_test[0]) == 4)
assert(len(y_train) == 80)
assert(len(y_test) == 20)

# set classifiers
h_set = set()
s_set = {1, -1}
b_set = set(np.linspace(0, 10, 51))
d_set = {0, 1, 2, 3}
for s in s_set:
	for b in b_set:
		for d in d_set:
			h_set.add(decision_stump.DecisionStump(s,b,d))

# training
Adas = []
Logs = []
for idx, T in enumerate([10, 20, 30]):
	Adas.append(boosting.AdaBoost(h_set, T=T))
	Adas[idx].train(X_train, y_train)

	Logs.append(boosting.LogitBoost(h_set, T=T))
	Logs[idx].train(X_train, y_train)

# testing
Ada_preds = []
Log_preds = []
Ada_accus = []
Log_accus = []
for Ada in Adas:
	pred = Ada.predict(X_test)
	Ada_preds.append(pred)
	Ada_accus.append(accuracy_score(pred, y_test))
for Log in Logs:
	pred = Log.predict(X_test)
	Log_preds.append(pred)
	Log_accus.append(accuracy_score(pred, y_test))
print('AdaBoost testing accuracies:', Ada_accus)
print('LogitBoost testing accuracies:', Log_accus)

# save 
json.dump({'Ada_preds': Ada_preds, 'Ada_accus': Ada_accus,
			'Log_preds': Log_preds, 'Log_accus': Log_accus},
			open('boosting.json', 'w'))