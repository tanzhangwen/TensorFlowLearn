import tensorflow as tf

# tf.get_variables(name, shape, initializer)
a = tf.get_variable('a', shape=[2,3], initializer = tf.random_normal_initializer(mean=0, stddev=1))
b = tf.get_variable('b', [1], initializer = tf.constant_initializer(1))
c = tf.get_variable('c', [2, 3], initializer = tf.ones_initializer())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(a)
	print(sess.run(a))
	print(sess.run(b))
	print(sess.run(c))