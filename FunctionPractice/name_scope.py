import tensorflow as tf

# name_scope works for Variable function
# variable_scope works for get_variable function
with tf.name_scope("V1"):
	a1 = tf.Variable(tf.random_normal(shape=[3,2],mean=0,stddev=1), name='a')
	
with tf.name_scope("V2"):
	a2 = tf.Variable(tf.random_normal(shape=[3,2],mean=0,stddev=1), name='a')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(a1.name)
	print(sess.run(a1))
	print(a2.name)
	print(sess.run(a2))