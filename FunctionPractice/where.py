import tensorflow as tf
import numpy as np

A = 3
B = tf.convert_to_tensor([1,2,3,4])
C = tf.convert_to_tensor([1,1,1,1])
D = tf.convert_to_tensor([0,0,0,0])

# where is used to replace select
with tf.Session() as sess:
	print(sess.run(tf.where(A > 1, 'B', 'C')))
	print(sess.run(tf.where(false, 'B', 'C')))
	print(sess.run(tf.where(B > 2, C, D)))