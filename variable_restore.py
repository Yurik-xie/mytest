import tensorflow as tf
import numpy as np

 # bulid the holder for 'w' and 'b'
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

 # 不需要对变量初始化：global_variables_initializer
saver = tf.train.Saver()

with tf.Session() as sess:
	 # restore variable 'w' and 'b'
	saver.restore(sess,"my_nn_variables/save_net.ckpt")
	print("weights:", sess.run(W))
	print("biases:", sess.run(b))