import tensorflow as tf

a = tf.Variable(tf.constant(1, shape=[2,2], dtype=tf.int32), name='a')
b = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='b')

variables = tf.global_variables()

print(variables[0].name, variables[0].value)
print(variables[1].name)