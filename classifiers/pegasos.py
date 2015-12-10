import numpy as np
import math

class Pegasos(object):
	"""docstring for Pegasos"""
	def __init__(self, reg, k, maxiter = 1000000, X=None, Y=None, check = False ):
		super(Pegasos, self).__init__()
		self.reg = reg
		self.k = k
		self.W = None
		self.iteration = 2
		self.maxiter = maxiter
		self.X = X
		self.Y = Y

		self.check = check

	def update(self, X, Y):
		if self.iteration > self.maxiter:
			return
		x_shape = X.shape
		y_shape = Y.shape

		assert len(y_shape) == 1, "This is a binary svm. Do ova or ovo method instead of multiclass."

		if self.W is None:
			self.W = 0.1/math.sqrt(self.reg) * np.random.randn(x_shape[1] , 1)

		#Learning Rate
		eta = 1.0 / (self.reg * self.iteration)

		#Evaluation
		p = np.multiply(np.dot(X, self.W), Y)

		

		#Get the support
		support = Y[0] * X
		if p[0][0] < 1:
			support = 0 * X

		#Stochastic Gradient Descent Step
		self.W = (1.0 - eta * self.reg) * self.W + eta * support.T / self.k 


		#Projection Step
		assert np.linalg.norm(self.W) != 0, "You can't divide zero"

		projection = 1.0 / math.sqrt(self.reg) / (np.linalg.norm(self.W))
		if projection < 1.0:
			self.W = projection * self.W

		if self.check:
			#Sanity Check on Loss Function
			if self.iteration % 1000 == 0:
				print "loss", self.loss(self.X, self.Y)
		

		self.iteration = self.iteration + 1

	def solve(self):
		pass

	def score(self, X, Y):
		print np.dot(X, self.W).shape
		print Y.shape
		prediction = np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		print "f", prediction.shape
		print prediction[prediction > 0].shape
		return len(prediction[prediction > 0]) / float(len(prediction))
		

	def loss(self, X, Y):
		loss = 1 - np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		temp = np.zeros(loss.shape)
		return np.sum(np.maximum(loss, temp))


	def infer(self, X):
		return np.dot(X, self.W)
