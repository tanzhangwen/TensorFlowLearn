import tensorflow as tf
import matplotlib.pyplot as plt

image_raw = tf.gfile.FastGFile('pic/ai7.jpg', 'rb').read()

with tf.Session() as sess:
	image_data = tf.image.decode_jpeg(image_raw)
	image_data = tf.image.convert_image_dtype(image_data,dtype=tf.float32)
	crop = tf.image.resize_image_with_crop_or_pad(image_data, 50, 50)
	pad = tf.image.resize_image_with_crop_or_pad(image_data, 200, 200)
	
	plt.figure(1)
	plt.imshow(crop.eval())
	plt.figure(2)
	plt.imshow(pad.eval())
	plt.show()