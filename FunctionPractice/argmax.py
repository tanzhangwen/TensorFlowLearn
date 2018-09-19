import tensorflow as tf

A = [[1,3,4,5,6]]
B = [[1,3,5],[2,4,5]]

with tf.Session() as sess:
	print(sess.run(tf.argmax(A, 1)))  	# 1 is to get the horizontal max
	print(sess.run(tf.argmax(B, 0)))	# 0 is to get the vertical max