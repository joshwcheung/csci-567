import numpy as np
from typing import List
from abc import abstractmethod

class Classifier(object):
	def __init__(self):
		self.clf_name = None

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	@abstractmethod
	def predict(self, features: List[List[float]]) -> List[int]:
		return