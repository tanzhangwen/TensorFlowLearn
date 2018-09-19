import tensorflow as tf

a = tf.constant(2.0)
b = tf.constant(3.0)
c = tf.constant(5.0)

ad = tf.add(a, b)
mul = tf.multiply(b, c)

with tf.Session() as sess:
	result = sess.run([mul, ad])
	print(result)
	print(type(result))
	print(type(result[0]))