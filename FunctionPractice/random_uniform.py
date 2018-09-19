import tensorflow as tf
import numpy as np

# parameter: shape, minvalue, maxvalue, dtype
with tf.Session() as sess:
	print(sess.run(tf.random_uniform(
		(4,4),0,4,tf.float32)))