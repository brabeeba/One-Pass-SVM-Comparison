import mnist
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import classifiers 
from scipy.linalg import svd
import math

PEGASOS = 0
SGDQN = 1
ASGD = 2

data = mnist.read_data_sets("MNIST_data/", one_hot=True)

print data.train.images.shape
print data.train.labels.shape

train_image = data.train.images.copy()
train_label = data.train.labels.copy()

train_label[train_label == 0 ] = -1

test_image = data.test.images.copy()
test_label = data.test.labels.copy()

test_label[test_label == 0] = -1

classifier_type = 1

if classifier_type == SGDQN:
	reg = 1e-4
	model = classifiers.SGDQN(reg, 1e5, 10, X=train_image, Y=train_label, maxiter = 1e8, check = True)

	for x in xrange(0,50000):
		model.update(train_image[[x]], train_label[[x]])

	print "accuracy", model.score(test_image, test_label)

elif classifier_type == ASGD:
	reg = 1e-4
	model = classifiers.ASGD(reg, X=train_image, Y=train_label, maxiter = 1e8, check = True)

	for x in xrange(0,50000):
		model.update(train_image[[x]], train_label[[x]])

	print "accuracy", model.score(test_image, test_label)

elif classifier_type == PEGASOS:
	reg = 1e-4
	model = classifiers.Pegasos(1e-4, 1, X=train_image, Y=train_label, maxiter = 1e8, check = True)
	for x in xrange(0,50000):
		model.update(train_image[[x]], train_label[[x]])

	print "accuracy", model.score(test_image, test_label)




