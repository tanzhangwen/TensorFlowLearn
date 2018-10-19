import tensorflow as tf
import matplotlib.pyplot as plt

image_test = tf.gfile.FastGFile("pic/ai7.jpg", 'rb').read()

with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_test)
	img_data_byte = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
	
	plt.figure(1)
	plt.imshow(img_data_byte.eval())
	plt.show()
	
	encode_img_png = tf.image.encode_png(img_data_byte)
	with tf.gfile.GFile("pic/ai7.png", "wb") as f:
		f.write(encode_img_png.eval())