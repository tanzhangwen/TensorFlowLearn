import tensorflow as tf

v1 = tf.Variable(tf.constant(1, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2, shape=[1]), name='v2')

v = v1 + v2

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	saver.save(sess, "model/test.ckpt")