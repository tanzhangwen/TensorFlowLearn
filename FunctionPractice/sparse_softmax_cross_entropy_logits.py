import tensorflow as tf

y2 = tf.convert_to_tensor([[0, 0, 1, 0]], dtype=tf.int64)
x2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(y2, 1), logits = x2)

with tf.Session() as sess:
	print(sess.run(tf.argmax(y2, 0)))
	print(sess.run(tf.argmax(y2, 1)))
	print('c2:', sess.run(c2))


# https://blog.csdn.net/laolu1573/article/details/60138455
# NN's output
logits = tf.constant([
	[1, 2, 3.0],
	[1, 2, 3],
	[1, 2, 3]])

# true label
y_ = tf.constant([
	[0, 0, 1.0],
	[0, 0, 1],
	[0, 0, 1]])

# 1. softmax
# soft_max:e^1/(e^1+e^2+e^3)=0.09003057
y = tf.nn.softmax(logits)

# 2. cross_entropy
# cross_entropy: -1 * log(0.66524094) * 3= 1.2228
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Combine softmax and cross_entropy in one function
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

with tf.Session() as sess:
	print("step1: softmax result = ", sess.run(y))
	print("step2: cross entropy result = ", sess.run(cross_entropy))
	print("softmax_cross_entropy_with_logits result = ", sess.run(cross_entropy2))