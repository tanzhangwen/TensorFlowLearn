import tensorflow as tf
import matplotlib.pyplot as plt

image_jpg = tf.gfile.FastGFile("pic/ai7.jpg", "rb").read()

with tf.Session() as sess:
	image_data = tf.image.decode_jpeg(image_jpg)
	image_data_type = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
	resize_0 = tf.image.resize_images(image_data_type, (500,500), method=0)
	resize_1 = tf.image.resize_images(image_data_type, (500,500), method=1)
	resize_2 = tf.image.resize_images(image_data_type, (500,500), method=2)
	resize_3 = tf.image.resize_images(image_data_type, (500,500), method=3)
	
	plt.figure(0)
	plt.imshow(resize_0.eval())
	plt.figure(1)
	plt.imshow(resize_1.eval())
	plt.figure(2)
	plt.imshow(resize_2.eval())
	plt.figure(3)
	plt.imshow(resize_3.eval())
	plt.show()