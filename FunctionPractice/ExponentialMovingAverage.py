import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(tf.constant(0))

# decay= min（decay，（1+steps）/（10+steps））
# shadow_variable = decay * shadow_variable + (1-decay) * variable
ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_average = ema.apply([v1])

with tf.Session() as sess:
	#init = tf.initialize_all_variables()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	print(sess.run([v1, ema.average(v1)]))
	
	sess.run(tf.assign(v1, 5))
	sess.run(maintain_average)
	print(sess.run([v1, ema.average(v1)])) # decay=min(0.99, 1/10)=0.1, v1=0.1*0+0.9*5=4.5
	
	sess.run(tf.assign(step, 10000))
	sess.run(tf.assign(v1, 10))
	sess.run(maintain_average)
	print(sess.run([v1, ema.average(v1)])) # decay=min(0.99,(1+10000)/(10+10000))=0.99, v1=0.99*4.5+0.01*10=4.555
	
	sess.run(maintain_average)
	print(sess.run([v1, ema.average(v1)])) #decay=min(0.99,(1+10000)/(10+10000))=0.99, v1=0.99*4.555+0.01*10=4.6