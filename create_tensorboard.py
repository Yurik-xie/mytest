from __future__ import print_function
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activition_function=None):
	 # add one more layer and return the output of this layer
	 # define layer name 
	with tf.name_scope('layer'):
		 # define weights name
		with tf.name_scope('weights'): 
			Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
		 # define biases 
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b') 
		  # define Wx_plus_b 
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)	
		if activition_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activition_function(Wx_plus_b)
		return outputs

 # define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

 # add hidden layer	
layer1 = add_layer(xs, 1, 10, activition_function=tf.nn.relu)
 # add output layer
prediction = add_layer(layer1, 10, 1, activition_function=None)

 # the error between prediction and real data
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
							reduction_indices=[1]))


with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)	

sess = tf.Session()

 # tf.train.summaryWriter soon be de
writer = tf.summary.FileWriter("E:\\python_program\\machine_learing_tensorflow\\my_code",
 sess.graph)
 
 #initilizer variables 
init = tf.global_variables_initializer()
sess.run(init)