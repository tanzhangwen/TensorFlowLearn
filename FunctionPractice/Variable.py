import tensorflow as tf

# tf.Variable(initializer, name)
a = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a')
b = tf.Variable(tf.constant(1), name = 'b')
c = tf.Variable(tf.ones(shape=[2,3]), name='c')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(a))
	print(sess.run(b))
	print(sess.run(c))