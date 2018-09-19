import tensorflow as tf

with tf.variable_scope("V1") as scope:
	a1 = tf.get_variable(name='a', shape=[1], initializer=tf.constant_initializer(1))
	scope.reuse_variables()
	a2 = tf.get_variable('a')

with tf.variable_scope("V2") as scope:
	b1 = tf.get_variable(name='a', shape=[1], initializer=tf.constant_initializer(1))

with tf.variable_scope("V2", reuse=True) as scope:
	b2 = tf.get_variable('a')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(a1.name)
	print(sess.run(a1))
	print(a2.name)
	print(sess.run(a2))
	
	print(b1.name)
	print(sess.run(b1))
	print(b2.name)
	print(sess.run(b2))