# # # A simple classification code using neural networks in tensor flow

import tensorflow as tf
import numpy as np
import create_data
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.datasets

import plot_boundary

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
data_file = ""
data_params = np.empty([7,1])

# np.random.seed(0)
# be default moon dataset
train_X, train_y = sklearn.datasets.make_moons(2000, noise=0.20)
test_X, test_y = sklearn.datasets.make_moons(400, noise=0.20)
train_y = np.array(train_y).astype(dtype=np.uint8)
train_y =  (np.arange(2) == train_y[:, None]).astype(np.float32)
test_y = np.array(test_y).astype(dtype=np.uint8)
test_y =  (np.arange(2) == test_y[:, None]).astype(np.float32)

def get_data():
	global data_file, data_params
	#call the function which creates data
	# [data_file, data_params] = create_data.create_LSD(1000, -2, 1, -2, 4, -2, 4)
	# [data_file,data_params] = create_data.create_PLSD(1000,-2,1,2,-1,-2,4,-2,4)
	[data_file, data_params] = create_data.create_PPLSD(3000, 2, -1, -2, -1,-1/16,2, -4, 4, -4, 4)
	# unprocessed data with polynomials
	# [data_file, data_params] = create_data.create_PolySD(1000, [6,0,-33,0,45], -4, 4, -1, 70)
	# [data_file, data_params] = create_data.create_PolySD(3000, [1.2471,0.3800,-1.9518,-0.0160], -1.5, 1.6, -1.5,1.5)
	# processed data with polynomials
	our_coeff = [6, 0, -33, 0, 45]
	# our_coeff = [1, -5, 4, -6, 9]
	# [our_min_x, our_max_x, our_min_y, our_max_y] = create_data.process_PolySD(our_coeff)
	# [data_file, data_params] = create_data.create_PolySD(1000, our_coeff,
	# 													 our_min_x, our_max_x, our_min_y, our_max_y)
	# data_file,data_params = "Data/2D_circlesSD.csv", []

	CSV_COLUMN_NAMES = ['x1', 'x2', 'label']
	full_data = pd.read_csv(data_file, names=CSV_COLUMN_NAMES, header=0)
	full_data["label"] = full_data["label"].astype("int")
	data, target = full_data[['x1', 'x2']], full_data[['label']]
	# to convert all panadas.dataframe to ndarray
	full_data, data, target = full_data.values, data.values, target.values
	target = np.reshape(target,(len(target),))
	all_X = data

	# Convert into one-hot vectors
	num_labels = len(np.unique(target))
	all_Y = np.eye(num_labels)[target]
	return train_test_split(all_X, all_Y, test_size=0.30, random_state=RANDOM_SEED)

# comment the line below to use default moon dataset
train_X, test_X, train_y, test_y = get_data()

learning_rate = 0.001
training_epochs = 20
batch_size = 10
display_step = 1

# define neural network
num_input = len(train_X[1,:])
num_classes = len(train_y[1,:])

num_hidden = [6,6,6,6,6]
len_num_hidden = len(num_hidden)
num_all = np.concatenate((np.array([num_input]),num_hidden,np.array([num_classes])))

weights={}
for i in range(len_num_hidden+1):
	weights['W'+str(i+1)] = tf.Variable(tf.random_normal([num_all[i],num_all[i+1]]))
biases = {}
for i in range(len_num_hidden+1):
	biases['b'+str(i+1)] = tf.Variable(tf.random_normal([num_all[i+1]]))

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


def forwardprop(X, len_num_hidden):
	W1 = weights['W1']
	b1 = biases['b1']
	layer = tf.add(tf.matmul(X, W1), b1)
	# layer = tf.nn.sigmoid(layer)
	layer = tf.nn.relu(layer)
	for i in range(len_num_hidden-1):
		W= weights['W'+str(i+2)]
		b = biases['b'+str(i+2)]
		layer = tf.add(tf.matmul(layer, W), b)
		# layer = tf.nn.sigmoid(layer)
		layer = tf.nn.relu(layer)

	Wnum_layers = weights['W'+str(len_num_hidden+1)]
	bnum_layers = biases['b'+str(len_num_hidden+1)]
	final_layer = tf.add(tf.matmul(layer, Wnum_layers), bnum_layers)
	return final_layer

# Forward propagation
yhat    = forwardprop(X,len_num_hidden)

# Backward propagation
cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=yhat))
updates = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = updates.minimize(cost)

train_size = 2000

# Run stochastic gradient descent
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	for step in range(training_epochs * train_size // batch_size):
		offset = (step * batch_size)%train_size
		batch_data = train_X[offset:(offset+batch_size),:]
		batch_labels = train_y[offset:(offset+batch_size)]
		feed_dict_train = {X: train_X,
				   Y: train_y}
		sess.run([train_op, cost], feed_dict=feed_dict_train)

	# print(sess.run(weights['W1']))
	predict = tf.nn.softmax(yhat)
	prediction = tf.argmax(predict, 1)
	correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({X: test_X, Y: test_y}))

	weights_topass = sess.run(weights)
	biases_topass = sess.run(biases)
	eval_fun = lambda Z: prediction.eval(feed_dict={X:Z})

	# new_X = X.eval(session=sess)
	# print(sess.run(new_X))
	# tf.Print(new_X[1:3,:],[new_X[1:3,:]])

	plot_boundary.plot_forDNN(train_X, train_y, test_X, test_y, eval_fun)
