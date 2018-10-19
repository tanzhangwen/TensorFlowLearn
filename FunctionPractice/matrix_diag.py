import tensorflow as tf

A = [1, 2, 3] # [[1, 2, 3]]
B = tf.matrix_diag(A) # one more dimension, if A is 1 then B is 2, if A is 2 then B is 3
print(B.eval(session=tf.Session()))

B = tf.cast(B, tf.float32)
C = tf.matrix_inverse(B)
print(C.eval(session=tf.Session()))