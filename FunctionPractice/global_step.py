import tensorflow as tf;  
import numpy as np;  
import matplotlib.pyplot as plt;  
 
x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
w = tf.Variable(tf.constant(0.0))
 
global_steps = tf.Variable(0, trainable=False)
# decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 2, staircase=False)
loss = tf.pow(w*x-y, 2)
 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10):
		sess.run(train_step, feed_dict={x:np.linspace(1,2,10).reshape([10,1]), y:np.linspace(1,2,10).reshape([10,1])})
		print(sess.run(global_steps))
		print(sess.run(learning_rate),sess.run(w))
		print(sess.run(loss, feed_dict={x:np.linspace(1,2,10).reshape([10,1]), y:np.linspace(1,2,10).reshape([10,1])}))
