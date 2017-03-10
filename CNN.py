import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 # number 1 to 10 data
mnist = input_data.read_data_sets('E:/python_program/machine_learing_tensorflow/my_code/MINIST_data', one_hot=True)

def compute_accurcy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W): # x为输入的值eg：图片
	 # stride [1, x_movement水平方向跨度, y_movement竖直方向, 1]
	 # Must have strides[0] = strides[3]  步长，每一步的跨度
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	 # stride [1, x_movement, y_movement, 1]
	 # 压缩图片长和宽，使图片变厚
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) #28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1,28,28,1])
 # -1代表先不考虑输入的图片例子多少这个维度，
 # 后面的1是channel的数量，
 # 因为我们输入的图片是黑白的，
 # 因此channel是1，例如如果是RGB图像，那么channel就是3。
 # print(x_image.shape) # [n_samples,28,28,1]

 ## conv1 layer##
W_conv1 = weight_variable([5,5,1,32]) 
 # patch(每次取的大小) 5x5 黑白图片channel=1,输出32个features
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 =max_pool_2x2(h_conv1)					# output size 14x14x32

 ## conv2 layer##
W_conv2 = weight_variable([5,5,32,64])
 # # patch(每次取的大小) 5x5 in size 32,out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)   		#output size 7x7x64

 ## function1 layer##
 #pool2的输出，让图像变得更厚
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
 # [n_sample, 7,7,64] ->> [n_sample, 7*7*64] 
 # 将三维数据转换为一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

 ## function2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

 # the error between prediction and real data
 #loss函数（即最优化目标函数）选用交叉熵函数。
 # 交叉熵用来衡量预测值和真实值的相似程度，
 # 如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
								reduction_indices=[1]))   # loss
train_setp = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_setp, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
	if i%50 == 0 :
		print(compute_accurcy(mnist.test.images, mnist.test.labels))