import time
import classifiers
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

def benchmark1():
	epoches = 1

	time1 = []
	time2 = []
	time3 = []
	time4 = []

	for sample in [10000, 100000, 1000000, 10000000]:
		
		
		X, Y = make_classification(n_samples = sample, random_state = 1111)
		Y[Y==0] = -1
		reg = 1e-4
		model1 = classifiers.Pegasos(reg, 1, X=X, Y=Y, maxiter = 1e8)
		model2 = classifiers.SGDQN(reg, 1e5, 10, X=X, Y=Y, maxiter = 1e8, check = False)
		model3 = classifiers.ASGD(reg, X=X, Y=Y, maxiter = 1e8)
		model4 = classifiers.OLBFGS(reg, 10 , X=X, Y=Y, check = False, maxiter = 1e8)

		
		for epoch in xrange(1,epoches + 1):
			start = time.clock()
			for i in xrange(0, sample):
				model1.update(X[[i]], Y[[i]])
			end = time.clock()
			time1.append(end - start)

			print "Finished Pegasos in {0}".format(time1[-1])

			start = time.clock()
			for i in xrange(0, sample):
				model2.update(X[[i]], Y[[i]])
			end = time.clock()
			time2.append(end - start)

			print "Finished SGD-QN in {0}".format(time2[-1])
			
			start = time.clock()
			for i in xrange(0, sample):
				model3.update(X[[i]], Y[[i]])
			end = time.clock()
			time3.append(end - start)

			print "Finished ASGD in {0}".format(time3[-1])

			start = time.clock()
			for i in xrange(0, sample):
				model4.update(X[[i]], Y[[i]])
			end = time.clock()
			time4.append(end - start)

			print "Finish oLBFGS in {0}".format(time4[-1])
			

			print "finish at {0}".format(sample)
		

	N = len(time1)
	ind = np.arange(N)
	width = 0.35

	fig, ax = plt.subplots()

	ax.set_yscale('log')

	index = np.arange(N)
	bar_width = 0.2

	opacity = 0.4

	rects1 = plt.bar(index, time1, bar_width, alpha=opacity,color='b',label='Pegasos')

	rects2 = plt.bar(index + bar_width, time2, bar_width, alpha=opacity,color='r',label='SGD-QN')

	rects3 = plt.bar(index + 2*bar_width, time3, bar_width, alpha=opacity,color='g',label='ASGD')

	rects4 = plt.bar(index +  3*bar_width, time4, bar_width, alpha=opacity,color='y',label='oLBFGS')



	plt.xlabel('models')
	plt.ylabel('training times after 1 epoch')
	plt.title('Comparison of SVM')
	plt.xticks(index + bar_width, ('1e4', '1e5', '1e6', '1e7'))
	plt.legend(loc='upper left')

	plt.tight_layout()
	plt.show()

def benchmark2():
	epoches = [0.5 * x for x in xrange(1, 12)]
	

		

		


			


if __name__ == '__main__':
	benchmark1()