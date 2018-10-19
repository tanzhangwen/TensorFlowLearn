import tensorflow as tf

a = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='a')
b = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='b')

global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)
ema = tf.train.ExponentialMovingAverage(0.09, global_step)

for te in tf.trainable_variables():
	print(te.name)
for ge in tf.global_variables():
	print(ge.name)