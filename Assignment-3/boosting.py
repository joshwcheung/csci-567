import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		
		h = np.zeros(len(features))
		for clf, beta in zip(self.clfs_picked, self.betas):
			h += beta * np.array(clf.predict(features))
		h = [-1 if hn <= 0 else 1 for hn in h]
		return h

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		
		# Step 1
		N = len(labels)
		w = np.full(N, 1 / N)
		
		# Step 2
		for t in range(self.T):
			# Step 3 & 4
			epsilon = float("inf")
			for clf in self.clfs:
				h = clf.predict(features)
				error = np.sum(w * (np.array(labels) != np.array(h)))
				if error < epsilon:
					ht = clf
					epsilon = error
					htx = h
			self.clfs_picked.append(ht)
			
			# Step 5
			beta = 1 / 2 * np.log((1 - epsilon) / epsilon)
			self.betas.append(beta)
			
			# Step 6
			for n in range(N):
				if labels[n] == htx[n]:
					w[n] *= np.exp(-beta)
				else:
					w[n] *= np.exp(beta)
			
			# Step 7
			w /= np.sum(w)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		
		# Step 1
		N = len(labels)
		pi = np.full(N, 0.5)
		f = 0
		
		# Step 2
		for t in range(self.T):
			# Step 3
			z = ((np.array(labels) + 1) / 2 - pi) / (pi * (1 - pi))
			
			# Step 4
			w = pi * (1 - pi)
			
			# Step 5
			epsilon = float("inf")
			for clf in self.clfs:
				h = clf.predict(features)
				error = np.sum(w * (z - np.array(h)) ** 2)
				if error < epsilon:
					ht = clf
					epsilon = error
					htx = h
			self.clfs_picked.append(ht)
			
			# Step 6
			self.betas.append(0.5)
			f += 0.5 * np.array(htx)
			
			# Step 7
			pi = 1 / (1 + np.exp(-2 * f))
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	