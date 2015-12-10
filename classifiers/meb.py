import numpy as np
import math
import cvxopt
import cvxopt.solvers

class CVM(object):
	"""docstring for CVM"""
	def __init__(self, X, y, epsilon, kernel, reg = 1e6, probability = True, sample = 60, max_iter = 150, max_try = 10):
		super(CVM, self).__init__()
		self.X = X
		self.epsilon = epsilon
		self.kernel = kernel
		self.reg = reg
		self.y = y

		self.defaultValue = None

		self.number = X.shape[0]
		self.probability = probability
		self.sample = sample

		self.remain = [x for x in xrange(0, self.number)]

		self.coreset = []
		self.radius = 0.0
		self.center = {}
		
		self.cache = {}

		self.max_iter = max_iter
		self.max_try = max_try

		self.alpha = np.zeros( ( 2, 1 ), np.float )
		self.matrix = np.zeros((3,3), dtype = np.float)

		value = self.epsilon

		cvxopt.solvers.options['show_progress'] = False
		cvxopt.solvers.options['maxiters'] = 100
		cvxopt.solvers.options['abstol'] = self.epsilon ** 2
		cvxopt.solvers.options['reltol'] = self.epsilon ** 2
		cvxopt.solvers.options['feastol'] = 1e-12
		cvxopt.solvers.options['refinement'] = 2


	def newKernel(self, a, b): #maybe vectorize later
		
		if (a, b) in self.cache:
			return self.cache[(a,b)]
		elif (b, a) in self.cache:
			return self.cache[(b,a)]

		delta = 0
		if a == b:
			delta = 1
			if self.defaultValue == None:
				self.defaultValue = self.y[a]*self.y[b]*self.kernel(self.X[a], self.X[b]) +  self.y[a]*self.y[b] + delta/self.reg
				return self.defaultValue
			else:
				return self.defaultValue
		

		result = self.y[a]*self.y[b]*self.kernel(self.X[a], self.X[b]) +  self.y[a]*self.y[b] + delta/self.reg
		self.cache[(a,b)] = result
		self.cache[(b, a)] = result
		return result


	def distance_compute(self, x): #maybe vectorize later
		result = 0.0
		for key1 in self.center:
			for key2 in self.center:
				result = result + self.center[key1]*self.center[key2]*self.newKernel(key1, key2)
			result = result - 2*self.center[key1]*self.newKernel(key1, x)
		return result + self.newKernel(x, x)


	def initialize(self):
		if self.probability:
			first_point = np.random.randint(self.number) 
			self.coreset.append(first_point)
			self.center[first_point] = 1
			self.remain.remove(first_point)

			second_point = self.furthest_point(0)
			if second_point is None:
				print "There is only one point"
				return
			self.coreset.append(second_point)
			self.center[first_point] = 0.5
			self.center[second_point] = 0.5
			self.remain.remove(second_point)
			self.radius = self.newKernel(first_point, second_point)/2.0


		else:
			print "Too slow. Will implement in future."

	def train(self):
		if self.probability:
			self.initialize()
			self.iteration = 0
			tries = 0

			while tries < self.max_try and self.iteration < self.max_iter:
				factor = 1 + self.epsilon
				factor = factor ** 2

				new_point = self.furthest_point(factor * self.radius)
				if new_point == None and tries >= self.max_try:
					print "Finish training for maximum tries."
					break
				elif new_point == None:
					tries = tries + 1
				else:
					tries = 0
					self.coreset.append(new_point)
					self.solve_qp()
				self.iteration = self.iteration + 1
			print "Finish training for maximum iteration"


		else:
			print "Too slow. Will implement in future"

	def solve_qp(self):
		n = len(self.coreset)
		if n == 3:
			for x in xrange(0, n):
				for y in xrange(0, n):
					self.matrix[x, y] = self.newKernel(self.coreset[x], self.coreset[y])
		elif n > 3 and self.matrix.shape[0] != n:
			new_row = np.zeros((1, n))
			new_column = np.zeros((n-1, 1))

			for x in xrange(0, n):
				if x < n-1:
					new_column[x, 0] = self.newKernel(self.coreset[x], self.coreset[n-1])
				new_row[0, x] = self.newKernel(self.coreset[n-1] , self.coreset[x])
			self.matrix = np.hstack([self.matrix, new_column])
			self.matrix = np.vstack([self.matrix, new_row])

		M1 = cvxopt.matrix(self.matrix)
		c1 = cvxopt.matrix(0.0, (n, 1))
		M2 = cvxopt.matrix(0.0, (n,n))
		M2[::n+1] = -1.0
		c2 = cvxopt.matrix( 0.0, (n,1))
		M3 = cvxopt.matrix(1.0, (1, n))
		c3 = cvxopt.matrix(1.0)

		print "Solving QP at iteration {0}".format(self.iteration)

		sol = cvxopt.solvers.qp(M1,c1, M2,c2,M3, c3)
		print "solved"

		if sol['status'] != 'optimal':
			print "could not solve the exact solution"
		self.alpha = np.asarray(sol['x']).copy()


		self.radius = np.dot(self.alpha.T, self.matrix.diagonal())[0] - np.dot(np.dot(self.alpha.T, self.matrix), self.alpha)[0,0]

		for x in xrange(0, n):
			self.center[self.coreset[x]] = self.alpha[x]


	def random_id(self):
		if len(self.remain) == 0:
			return None
		else:
			return np.random.choice(self.remain)

	def predict(self, new):
		result = 0
		bias = 0
		n = len(self.coreset)
		for x in xrange(0, n):
			result = result + self.alpha[x]*self.y[self.coreset[x]]*self.kernel(self.X[self.coreset[x]], new)
			bias = bias + self.alpha[x]*self.y[self.coreset[x]]

		result = result + bias
		if result > 0:
			return 1
		else:
			return 0

	def furthest_point(self, distance):
		if self.probability:

			sample_list = []
			maximumIndex = None
			maximum = distance
			for x in xrange(0, self.sample):
				point = self.random_id()
				if point == None:
					break
				else:

					if maximum < self.distance_compute(point):
						maximum = self.distance_compute(point)
						maximumIndex = point

			return maximumIndex

		else:
			print "Too slow. Will implement in future."






		