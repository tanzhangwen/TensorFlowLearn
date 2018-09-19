import tensorflow as tf
import numpy as np

A = tf.Variable(tf.constant(0, dtype = tf.float32))
B = tf.Variable(tf.constant([0,0], dtype = tf.float32))

# initialize_all_variables deprecated
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(A))
	print(sess.run(B))
	sess.run(tf.assign(A, 10.0))
	sess.run(tf.assign(B, [2,10.0]))
	print(sess.run(A))
	print(sess.run(B))