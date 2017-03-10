import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##test1  start##

##creat data### 
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1 + 0.3

## creat tensorflow structure start ### 
# Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# biases = tf.Variable(tf.zeros([1]))

# y = Weights*x_data + biases

# loss = tf.reduce_mean(tf.square(y-y_data))	##均方差
# optimizer = tf.train.GradientDescentOptimizer(0.5) ##梯度下降优化器
# train = optimizer.minimize(loss) ##

# init = tf.global_variables_initializer()  ##Attention: variables rather than variable

## creat tensorflow structure end ###

# sess = tf.Session()
# sess.run(init)     #activate the initiation

# for step in range(201):
	# sess.run(train)
	# if step % 20 == 0:
		# print(step,sess.run(Weights),sess.run(biases))
		
##test1  end##






##test2  start  about matrix multiply##

####creat two matrix
# matrix1 = ([[3,3]])
# matrix2 = ([[2],
			# [2]])

# product = tf.matmul(matrix1,matrix2)  
####product不直接计算，我们需要Session来激活product

####method 1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
#####  method  1  simplify
# print(tf.Session().run(product))
# tf.Session().close()

#####method 2   use with 
# with tf.Session() as sess:
	# result2 = sess.run(product)
	# print(result2)

# state = tf.Variable(0,name='counter')

# ### define constant one
# one = tf.constant(1)

# #### define addition procedure Attation: this step does't caculator variable
# new_value = tf.add(state, one)

# ### update State into new_value
# update = tf.assign(state, new_value)

# ### if define the Variable ,you must initialize it
# init = tf.global_variables_initializer()

# ### use Session
# with tf.Session() as sess:
# 	sess.run(init)
# 	for _ in range(3):
# 		sess.run(update)
# 		print(sess.run(state))

# ##test2  end##







# ###test3  start  Placeholder ##
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)

# ###calculator input1 multiply input2 and assignment the result to output
# output = tf.multiply(input1,input2)

# with tf.Session() as sess:
# 	print(sess.run(output,feed_dict={input1:[5.1],input2:[3.]}))
# ##test3  end##





# ###test4  start  def add_layer() ##
def add_layer(inputs, in_size, out_size, activition_function=None):
	#### Weights is a matrix  random variable row=in_size column=out_size
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	#### biases should not be zero
	biases = tf.Variable(tf.zeros([1, out_size])  + 0.1)
	#### define Wx_plus_b 即神经网络未激活的值 tf.matmul() matrix multiply
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	#### 当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，
	#### 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
	if activition_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activition_function(Wx_plus_b)

	return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

	### tf.placeholder() 代表占位符，利用占位符定义我们需要的神经网络的输入
	### None表示无论输入多少都可以，1表示输入特征只有一个
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

	### 定义隐藏层
layer1 = add_layer(xs, 1, 10, activition_function=tf.nn.relu)

 # 定义输入层 此时的输入是隐藏层的输出11，输入有10层（隐藏层的输出层），输出有1层
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
##test4  end ##




















