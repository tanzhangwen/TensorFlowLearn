import tensorflow as tf
import numpy as np

# check the value can be divided by 3 or 5
# 0 if not for both 3 and 5
# 1 for 3
# 2 for 5
# 3 for both 3 and 5
def ValueTag(value):
	if value%3==0 and value%5==0:
		return 3
	if value % 5 == 0:
		return 2
	if value % 3 == 0:
		return 1
	return 0

# generate the 10 bits representation of the values and 4 bits of the type it will be divided by 3 or 5 between [low, up]
def FizzBuzz(low, up):
	assert low < up
	x = []
	y = []
	for i in range(low, up+1):
		xx = [i >> d & 1 for d in range(10)]
		'''
		xx = [0 for t in range(10)]
		j = 9
		t = i
		while t:
			xx[j] = t % 2
			t //= 2
			j = j - 1
		'''
		yy = [ 1 if t == ValueTag(i) else 0 for t in range(0, 4)]
		x.append(xx)
		y.append(yy)
		return np.array(x),np.array(y)

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.02))

def model(X, w_h, w_o):
	#h = tf.nn.relu(tf.matmul(X, w_h))
	h = tf.nn.sigmoid(tf.matmul(X, w_h))
	return tf.matmul(h, w_o)

# Input and output
X = tf.placeholder("float", [None, 10])
Y = tf.placeholder("float", [None, 4])

NUM_HIDDEN = 100
w_h = init_weights([10, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

py_x = model(X, w_h, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=py_x))
#train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
train_op = tf.train.AdagradOptimizer(0.1).minimize(cost)

predict_op = tf.argmax(py_x, 1)

BATCH_SIZE = 128
x_train,y_train = FizzBuzz(101, 200)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	
	for epoch in range(100):
		# Shuffle the data before each training iteration
		p = np.random.permutation(range(len(x_train)))
		x_train, y_train = x_train[p], y_train[p]
		
		# Train in batches of batch size
		for start in range(0, len(x_train), BATCH_SIZE):
			end = start + BATCH_SIZE
			sess.run(train_op, feed_dict={X: x_train[start:end], Y: y_train[start:end]})
		
		# Print current accuracy on the training data
		print(epoch, sess.run(cost, feed_dict={X:x_train, Y:y_train}))
		print(epoch, np.mean(np.argmax(y_train, axis=1) == sess.run(predict_op, feed_dict={X:x_train, Y:y_train})))
	
	# Do testing
	x_test,y_test = FizzBuzz(1, 100)
	y_predict = sess.run(predict_op, feed_dict={X:x_test})
	print("Predict accuracy:", np.mean(np.argmax(y_test, axis=1) == y_predict))


