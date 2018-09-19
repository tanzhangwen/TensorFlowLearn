import tensorflow as tf
import numpy as np

A = [[1,2,3],[2,3,4]]
B = [[1,3,2],[3,2,4]]

with tf.Session() as sess:
	print(sess.run(tf.equal(A, B)))
	
