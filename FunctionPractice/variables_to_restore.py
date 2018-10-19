import tensorflow as tf

v = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='v')

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.all_variables())

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.assign(v, 10.0))
	sess.run(maintain_average_op)
	saver.save(sess, 'model/res.ckpt')
	print(sess.run(v))
	#print(sess.run(ema.average(v)))
	print(sess.run(v))

saver2 = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
	saver2.restore(sess, 'model/res.ckpt')
	print(sess.run(v))