import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1, shape=[1], dtype=tf.float32), name = 'v1')
v2 = tf.Variable(tf.constant(2, shape=[1], dtype=tf.float32), name='v2')
result = v1+v2

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	graph_def = tf.get_default_graph().as_graph_def()
	output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
	with tf.gfile.GFile('model/graph.pb', 'wb') as f:
		f.write(output_graph_def.SerializeToString())

with tf.Session() as sess:
	model_filename='model/graph.pb'
	with tf.gfile.FastGFile(model_filename, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		result = tf.import_graph_def(graph_def, return_elements=['add:0'])
		print(sess.run(result))