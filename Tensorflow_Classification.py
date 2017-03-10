import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('E:\\python_program\\machine_learing_tensorflow\\my_code\\MINIST_data',
						 one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None,):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b,)
	return outputs

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_prediction = sess.run(prediction, feed_dict={xs: v_xs})
	correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
	return result

 # define placeholder for inout to network
xs = tf.placeholder(tf.float32, [None, 784]) #训练集中每张图片的分辨率28*28
ys = tf.placeholder(tf.float32, [None, 10])  #每张定义一个数字，输出为0-9

 # add inout layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax) # 分类常用的激活函数

 # the error between prediction and real data
cross_enrtopy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
								reduction_indices=[1]))  #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_enrtopy)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))  