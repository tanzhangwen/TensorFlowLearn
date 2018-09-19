import tensorflow as tf

# truncated_normal(shape, mean, stddev) -> nomal distribution with value-mean <= 2*stddev
c = tf.truncated_normal([10,10], 0, 1)

with tf.Session() as sess:
	print(sess.run(c))