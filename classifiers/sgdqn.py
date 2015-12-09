import numpy as np
import math

class SGDQN(object):
	"""docstring for SGDQN"""
	def __init__(self, reg, t0, skip, maxiter = 1000000, X = None, Y = None):
		super(SGDQN, self).__init__()
		self.reg = reg
		self.t0 = t0
		self.maxiter = maxiter
		self.skip = skip

		self.iteration = 0
		self.count = skip
		self.updateB = False
		self.r = 2

		self.B = None
		self.W = None
		self.V = None

		self.X = X
		self.Y = Y

	def update_flaw(self, X, Y):
		x_shape = X.shape
		y_shape = Y.shape

		if self.B is None:
			self.B = np.empty(x_shape[1])
			self.B.fill(1.0 / self.reg)

		if self.W is None:
			self.W = np.zeros((x_shape[1], 1))

		if self.V is None:
			self.V = np.zeros((x_shape[1], 1))

		self.W = self.W - 1.0 / (self.iteration + self.t0) * self.dloss(np.multiply(np.dot(X, self.W), Y)[0][0]) * Y[0] * np.multiply(self.B, X)

		if self.updateB:
			diff_loss = self.dloss(np.multiply(np.dot(X, self.W), Y)[0][0]) - self.dloss(np.multiply(np.dot(X, self.V), Y)[0][0])

			r = np.zeros(x_shape[1])
			if diff_loss == 0:
				r.fill(self.reg)
			else:
				r = self.reg - diff_loss / self.dloss(self.z) * np.reciprocal(self.B)
				r[X.reshape(x_shape[1]) == 0] = self.reg
			temp1 = np.empty(x_shape[1])
			temp1.fill(self.reg)

			r = np.minimum (r, temp1 * 100)
			self.B = self.B + 2.0 / self.r * (np.reciprocal(r) - self.B)
			self.r = self.r + 1
			self.updateB = False

		self.count = self.count - 1

		if self.count <= 0:
			self.count = self.skip
			self.updateB = True
			self.W = self.W - self.skip * self.reg / (self.iteration + self.t0) * np.multiply(self.B, self.W)
			self.V = self.W

		#Sanity Check on Loss Function
		if self.iteration % 1000 == 0:
			print "loss", self.loss(self.X, self.Y)

		self.iteration = self.iteration + 1



	def update_corrected(self, X, Y):
		x_shape = X.shape
		y_shape = Y.shape

		if self.B is None:
			self.B = np.empty(x_shape[1])
			self.B.fill(1.0/(self.reg * self.t0))

		if self.W is None:
			self.W = np.zeros((x_shape[1], 1))

		if self.V is None:
			self.V = np.zeros((x_shape[1], 1))

		if self.updateB:
			diff_loss = self.dloss(np.multiply(np.dot(X, self.W), Y)[0][0]) - self.dloss(np.multiply(np.dot(X, self.V), Y)[0][0])

			r = np.zeros(x_shape[1])
			if diff_loss == 0:
				r.fill(self.reg)
			else:
				r = self.reg - diff_loss / self.dloss(self.z) * np.reciprocal(self.B)
				r[X.reshape(x_shape[1]) == 0] = self.reg

			temp1 = np.empty(x_shape[1])
			temp1.fill(self.reg)

			r = np.maximum(np.minimum (r, temp1 * 100), temp1)

			self.B = self.B / (1 + self.skip * np.multiply(self.B, r))
			self.updateB = False 
			

		self.z = np.multiply(np.dot(X, self.W), Y)[0][0]
		self.count = self.count - 1

		if self.count < 0:
			self.count = self.skip
			self.updateB = True
			self.V = self.W
			self.W = self.W - self.skip * self.reg * np.multiply(self.B, self.W)

		self.W = self.W - self.dloss(self.z) * Y[0] * np.dot(X, np.diag(self.B)).T

		
		#Sanity Check on Loss Function
		if self.iteration % 1000 == 0:
			print "loss", self.loss(self.X, self.Y)
		
		
		self.iteration = self.iteration + 1

	def dloss(self, z):
		#This is only for L1-SVM now. Todo: Implement more loss function
		if z < 1:
			return 1
		return 0
	
	def loss(self, X, Y):
		loss = 1 - np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		return np.sum(loss[loss > 0])

	def solve(self):
		pass


	def infer(self, X):
		return np.dot(X, self.W)

		


	 	


