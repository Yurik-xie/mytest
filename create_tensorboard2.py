import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activition_function=None):
	 # add one more layer and return the output of this layer
	layer_name = 'layer%s'%n_layer

	with tf.name_scope('layer'):
		with tf.name_scope('weights'): 
			Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
			tf.summary.histogram(layer_name + '/weights', Weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
			tf.summary.histogram(layer_name + '/biases', biases) 
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)	
		if activition_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activition_function(Wx_plus_b)

		tf.summary.histogram(layer_name + '/outputs', outputs)

		return outputs

 # define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

 # make up some data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data)- 0.5 + noise


 # add hidden layer	
layer1 = add_layer(xs, 1, 10, n_layer=1, activition_function=tf.nn.relu)
 # add output layer
prediction = add_layer(layer1, 10, 1, n_layer=2, activition_function=None)

 # the error between prediction and real data
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
							reduction_indices=[1]))
	tf.summary.scalar('loss', loss)


with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)	

sess = tf.Session()

merged = tf.summary.merge_all()

 # tf.train.summaryWriter soon be de
writer = tf.summary.FileWriter("E:\\python_program\\machine_learing_tensorflow\\my_code\\logs",
 sess.graph)
 
 #initilizer variables 
init = tf.global_variables_initializer()
sess.run(init)


for i in range(1000):
   sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
   if i%50 == 0:
      rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
      writer.add_summary(rs, i)
