import tensorflow as tf
import numpy as np

a = np.random.random([10,5])
b = tf.nn.embedding_lookup(a, [1, 3])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(b))
	print("----")
	print(a)