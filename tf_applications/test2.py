from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np



np.random.seed(10)
x = np.random.rand(10)
print x

y = np.random.rand(10)
print y

colors = np.random.rand(10)

plt.scatter(x, y, c=colors)
plt.xlabel("qwerty")
plt.ylabel("asdfgh")
plt.show()


W = tf.Variable(tf.random_normal([2, 1], name='weight'))


#with tf.Session() as sess:
#	sess.run(tf.initialize_all_variables())
#	print W
#	print sess.run(W)


