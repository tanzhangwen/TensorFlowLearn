import tensorflow as tf

reader = tf.train.NewCheckpointReader("model/test.ckpt")

variables = reader.get_variable_to_shape_map()

for ele in variables:
	print(ele)