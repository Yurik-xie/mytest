import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activition_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size])  + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if activition_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activition_function(Wx_plus_b)

	return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

	
layer1 = add_layer(xs, 1, 10, activition_function=tf.nn.relu)

prediction = add_layer(layer1, 10, 1, activition_function=None)

 # 计算误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
	reduction_indices=[1]))

 # 设置learning rate =0.1 meaning: 以0.1的效率来最小化误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
 # initialize all variables after define variables MUST 
init = tf.global_variables_initializer()

 # run initialization only the Sesson.run() will make it run in tensorflow
sess = tf.Session()
sess.run(init)

 # plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
 # plt.ion() # 运行show时，如果不使用ion，程序会停在此处
 # plt.show()

 # start training 
for i in range(1000):
	#training
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
	if i%50 == 0:
		# print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
		try:
			ax.lines.remove(line[0])
		except Exception :
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:x_data, ys: y_data})
		 # plot the prediction
		line = ax.plot(x_data, prediction_value, 'r-', lw=5)
		plt.pause(0.1) # 绘制完一次图，停顿0.1s
	if i ==999:
		plt.show()  # 停在最后一次输出结果上
	else:
		pass	
