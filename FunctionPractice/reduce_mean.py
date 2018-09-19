import tensorflow as tf
import numpy as np

A = np.random.uniform(0, 10, 12).reshape((3,4))

# get the mean value
h_mean = tf.reduce_mean(A, 1)
v_mean = tf.reduce_mean(A, 0)
mean = tf.reduce_mean(A)

print(A)

with tf.Session() as sess:
	print(sess.run(h_mean))
	print(sess.run(v_mean))
	print(sess.run(mean))