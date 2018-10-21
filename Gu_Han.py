import numpy as np
import create_data
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import plot_boundary

from sklearn.model_selection import train_test_split
import sklearn.datasets
# this is the implementation of scheme from "Clustered Support Vector Machines" paper from Gu and Han
# Gu, Q., Han, J.: Clustered support vector machines. In: Artificial Intelligence and Statistics, pp. 307–315 (2013)

RANDOM_SEED = 42

def get_data():
	global data_file, data_params
	#call the function which creates data
	# [data_file, data_params] = create_data.create_LSD(1000, -1, 1, -4, 4, -2, 4)
	# [data_file,data_params] = create_data.create_PLSD(1000,-2,1,2,-1,-2,4,-2,4)
	# [data_file, data_params] = create_data.create_PPLSD(3000, 2, -1, -2, -1,-1/16,2, -4, 4, -4, 4)
	# [data_file, data_params] = create_data.create_PolySD(1000, [6,0,-33,0,45], -4, 4, -1, 70)
	CSV_COLUMN_NAMES = ['x1', 'x2', 'label']
	full_data = pd.read_csv(data_file, names=CSV_COLUMN_NAMES, header=0)
	full_data["label"] = full_data["label"].astype("int")
	data, target = full_data[['x1', 'x2']], full_data[['label']]
	# to convert all panadas.dataframe to ndarray
	full_data, data, target = full_data.values, data.values, target.values
	target = 2*target-1
	all_X = data
	return train_test_split(all_X, target, test_size=0.33, random_state=RANDOM_SEED)

def cluster_data(X,num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1300).fit(X)
    return kmeans

# train_X, test_X, train_y, test_y = get_data()

# for the moon data set
train_X, train_y = sklearn.datasets.make_moons(2000, noise=0.20)
test_X, test_y = sklearn.datasets.make_moons(400, noise=0.20)
train_y = np.array(train_y).astype(dtype=np.float32)
test_y = np.array(test_y).astype(dtype=np.float32)
train_y = 2*train_y-1
test_y = 2*test_y-1

train_size = train_X.shape[0]
test_size = test_X.shape[0]
num_clusters = 4
train_y = np.reshape(train_y,(train_y.shape[0],))

train_cluster = cluster_data(train_X,num_clusters)

#creation of \tilde(x_i^l) (everything is only valid for 2D)
train_X_tilde = np.zeros([train_size,2*(num_clusters+1)])
lamda = 2
for i in range(train_size):
	this_X = train_X[i,:]
	train_X_tilde[i,0:2] = 1/np.sqrt(lamda)*this_X
	temp = (train_cluster.predict([this_X]))[0]
	train_X_tilde[i,temp*2+2:temp*2+4] = this_X

clf = LinearSVC(random_state=0)
clf.fit(train_X_tilde,train_y)
# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#      verbose=0)

w_tilde = clf.coef_
bias = clf.intercept_
w_tilde = np.reshape(w_tilde,(w_tilde.shape[1],1))
w = 1/np.sqrt(lamda)*w_tilde[0:2]

v_s ={}
w_s = {}
for i in range(num_clusters):
	temp = w_tilde[(i+1)*2:(i+1)*2+2]
	v_s['v' + str(i + 1)] = temp
	w_s['w'+str(i+1)] = temp + w

train_indices = {}
for i in range(num_clusters):
	train_indices['indices'+str(i+1)] = np.where(train_cluster.predict(train_X)==i)


plot_boundary.plot_forGuHan(train_X, train_y, train_cluster, train_indices, w_s, bias)

# keep this line always here for plotting
plt.show(block=True)
