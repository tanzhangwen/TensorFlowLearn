import tensorflow as tf
import numpy as np

A = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])

# tf.clip_by_value(A, min, max) will replace the values less than min with min and more than max with max in A 
with tf.Session() as sess:
	print(sess.run(tf.clip_by_value(A, 3, 5)))