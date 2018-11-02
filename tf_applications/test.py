from sklearn.utils import shuffle
from matplotlib import pyplot as plt 
import tensorflow as tf
import numpy as np


def generate(sample_size, mean, cov, diff, regression):
	num_classes = 2
	samples_per_class = int(sample_size/2)

	X0 = np.random.multivariate_normal(mean, cov, samples_per_class)  #samples_per_class=500
	#print X0
	
	Y0 = np.zeros(samples_per_class)
	#print Y0

	for ci, d in enumerate(diff):
		X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
		Y1 = (ci+1)*np.ones(samples_per_class)

		X0 = np.concatenate((X0, X1))
		Y0 = np.concatenate((Y0, Y1))

	print X0
	print Y0
	
	if regression==False:
		class_ind = [Y==class_number for class_number in range(num_classes)]
		Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
	X, Y = shuffle(X0, Y0)
	print X
	print Y
	return X,Y


np.random.seed(10)
mean = np.random.randn(2)
cov = np.eye(2)

X, Y = generate(10, mean, cov, [3.0], True)




