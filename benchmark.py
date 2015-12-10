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
		model2 = classifiers.SGDQN(reg, 1e5, 10, X=X, Y=Y, maxiter = 1e8, check = True)
		model3 = classifiers.ASGD(reg, X=X, Y=Y, maxiter = 1e8, check = False)
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
	epoch = 6
	epoches = [0.5 * x for x in xrange(1, epoch * 2 + 1)]
	sample = 100000
	X, Y = make_classification(n_samples = sample, random_state = 1111)
	Y[Y==0] = -1
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

	train_number = len(X_train)
	test_number = len(X_test)

	model1 = classifiers.Pegasos(1e-2, 1, X=X_train, Y=Y_train, maxiter = 1e8)
	model2 = classifiers.SGDQN(1e-4, 1e5, 10, X=X_train, Y=Y_train, maxiter = 1e8, check = False)
	model3 = classifiers.ASGD(1e-4, X=X_train, Y=Y_train, maxiter = 1e8)
	model4 = classifiers.OLBFGS(1e-4, 10 , X=X_train, Y=Y_train, check = False, maxiter = 1e8)

	score1 = []
	score2 = []
	score3 = []
	score4 = []

	for x in xrange(1, epoch * train_number + 1):
		randomID = np.random.randint(0, train_number - 1)

		model1.update(X_train[[randomID]], Y_train[[randomID]])
		model2.update(X_train[[randomID]], Y_train[[randomID]])
		model3.update(X_train[[randomID]], Y_train[[randomID]])
		#model4.update(X_train[[randomID]], Y_train[[randomID]])



		if x % int(sample / 2.0) == 0:
			score1.append(model1.score(X_test, Y_test))
			print "Pegasos accuracy {:10.4f} at {:.2f} epoches".format(score1[-1], float(x)/sample)

			score2.append(model2.score(X_test, Y_test))
			print "SGDQN accuracy {:10.4f} at {:.2f} epoches".format(score2[-1], float(x)/sample)

			score3.append(model3.score(X_test, Y_test))
			print "ASGD accuracy {:10.4f} at {:.2f} epoches".format(score3[-1], float(x)/sample)

			#score4.append(model4.score(X_test, Y_test))
			#print "oLBFGS accuracy {:10.4f} at {:.2f} epoches".format(score4[-1], float(x)/sample)


	plt.plot(epoches, score1, color = 'b', label='Pegasos')
	plt.plot(epoches, score2, color='r', label='SGD-QN')
	plt.plot(epoches, score3, color = 'g', label = 'ASGD')
	plt.plot(epoches, score4, color = 'y', label = 'oLBFGS')
	plt.xlabel('epoches')
	plt.ylabel('accuracy')
	plt.title('Epoches to Accuracy')
	plt.legend()
	plt.show()







if __name__ == '__main__':
	#benchmark1()
	benchmark2()