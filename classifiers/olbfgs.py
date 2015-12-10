import numpy as np
import math

class OLBFGS(object):
	"""docstring for OLBFGS"""
	def __init__(self, reg, m, epsilon = 1e-10, rho = 100.0, eta0 = 1.0, maxiter = 1000000, check = False, X = None, Y = None):
		super(OLBFGS, self).__init__()

		self.reg = reg
		self.m = m

		self.epsilon = epsilon
		self.rho = rho
		self.eta0 = eta0

		self.maxiter = maxiter
		self.check = check

		self.B = None
		self.W = None

		self.iteration = 0

		self.s = []
		self.y = []

		self.X = X
		self.Y = Y

	def update(self, X, Y):
		if self.iteration > self.maxiter:
			return
		x_shape = X.shape
		y_shape = Y.shape

		assert len(y_shape) == 1, "This is a binary svm. Do ova or ovo method instead of multiclass."

		if self.W is None:
			self.W = 0.01/math.sqrt(self.reg) * np.random.randn(x_shape[1] , 1)

		if self.B is None:
			temp = np.empty(x_shape[1])
			temp.fill(self.epsilon)
			self.B = np.diag(temp)

		p = None

		if self.iteration == 0:
			z = np.multiply(np.dot(X, self.W), Y)[0][0]
			p = -1.0 * self.epsilon * np.dot(self.B, self.dloss(z) * Y[0] * X.T).T


		else:
			z = np.multiply(np.dot(X, self.W), Y)[0][0]
			gradient = self.dloss(z) * Y[0] * X

			p = self.direction(gradient)

		eta = self.rho / (self.rho + self.iteration) * self.eta0


		self.s.append(eta * p)


		if self.iteration >= self.m:
			del self.s[0]

		z0 = np.multiply(np.dot(X, self.W), Y)[0][0]

		self.W = self.W + self.s[-1]

		z1 = np.multiply(np.dot(X, self.W), Y)[0][0]

		self.y.append(self.dloss(z1) * Y[0] * X - self.dloss(z0) * Y[0] * X + self.reg * self.s[-1])

		if self.iteration >= self.m:
			del self.y[0]


		if self.check:
			#Sanity Check on Loss Function
			if self.iteration % 1000 == 0:
				print "loss", self.loss(self.X, self.Y)
		

		self.iteration = self.iteration + 1

	def solve(self):
		pass

	def direction(self,gradient):
		p = -gradient


		number = min(self.iteration, self.m)

		a = []
		b = 0

		for x in xrange(1,number + 1):

			tempS = np.asarray(self.s[-x], dtype = "float32")
			tempY = np.asarray(self.y[-x], dtype = "float32")

			temp = np.dot(tempS, tempY.T)[0][0]
			if temp == 0:
				temp = 1e-10
			
			a.append(np.dot(p, tempS.T)[0][0] / temp)
			p = p - a[x - 1] * tempY


		tempS = np.asarray(self.s[-1], dtype = "float32")
		tempY = np.asarray(self.y[-1], dtype = "float32")

		temp = np.dot(tempY, tempY.T)[0][0]
		if temp == 0:
			temp = 1e-10
		p = np.dot(tempS, tempY.T)[0][0] / temp  * p


		for x in xrange(0,number):

			tempY = np.asarray(self.y[x], dtype = "float32")
			tempS = np.asarray(self.s[x], dtype = "float32")

			temp = np.dot(tempY, tempS.T)[0][0]
			if temp == 0:
				temp = 1e-10

			b = np.dot(tempY, p.T)[0][0] / temp
			p = p + (a[-(x+1)] - b) * tempS

		return p
	def score(self, X, Y):
		prediction = np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		return len(prediction[prediction > 0]) / float(len(prediction))
			

	def dloss(self, z):
		#This is only for L1-SVM now. Todo: Implement more loss function
		if z < 1:
			return -1
		return 0

	def loss(self, X, Y):
		loss = 1 - np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		temp = np.zeros(loss.shape)
		return np.sum(np.maximum(loss, temp))


	def infer(self, X):
		return np.dot(X, self.W)
