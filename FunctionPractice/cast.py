import tensorflow as tf
import numpy as np

A = tf.convert_to_tensor(np.array([[1,2,4,5],[3,5,5,6]], dtype=np.int32))

print(A)

with tf.Session() as sess:
	print(A.dtype)
	B = tf.cast(A, tf.float32)
	print(B.dtype)