import tensorflow as tf
import numpy as np

A = np.linspace(1, 20, 10)
B = tf.linspace(1.0, 20.0, 10)

print(A)
print(B.eval(session=tf.Session()))