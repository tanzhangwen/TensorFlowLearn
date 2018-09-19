import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
	print(sess.run(y, {x: [1.0, 2.0, 3.0]}))

y = tf.matmul([[37.0, -23.0],[1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
	options = tf.RunOptions()
	options.output_partition_graphs = True
	options.trace_level = tf.RunOptions.FULL_TRACE
	
	metadata = tf.RunMetadata()
	
	sess.run(y, options=options, run_metadata=metadata)
	
	#print(metadata.partition_graphs)
	
	print(metadata.step_stats)