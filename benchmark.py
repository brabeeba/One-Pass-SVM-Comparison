import time
import classifiers
import numpy

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

def benchmark():
	epoches = 1

	for sample in [10000, 100000, 1000000, 10000000]:
		
			
		X, Y = make_classification(n_samples = sample, random_state = 1111)
		Y[Y==0] = -1
		
		reg = 1e-4
		
		model1 = classifiers.Pegasos(reg, 1, X=X, Y=Y)
		model2 = classifiers.SGDQN(reg, 1e5, 10, X=X, Y=Y)
		model3 = classifiers.ASGD(reg, X=X, Y=Y)
		model4 = classifiers.OLBFGS(reg, 10 , X=X, Y=Y, check = True)


		start = time.clock()

		for epoch in xrange(1,epoches + 1):
			for i in xrange(0, sample):
				#model1.update(X[[i]], Y[[i]])
				#model2.update(X[[i]], Y[[i]])
				#model3.update(X[[i]], Y[[i]])
				model4.update(X[[i]], Y[[i]])

		end = time.clock()

		print end - start
		


			


if __name__ == '__main__':
	benchmark()