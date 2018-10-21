import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import random

from matplotlib import colors
def plot_forDNN(train_X, train_y, test_X, test_y, pred_func):
    # determine canvas borders
    mins = np.amin(train_X, 0)
    mins = mins - 0.1 * np.abs(mins)
    maxs = np.amax(train_X, 0)
    maxs = maxs + 0.1 * maxs

    ## generate dense grid
    xs, ys = np.meshgrid(np.linspace(mins[0], maxs[0], 1000), np.linspace(mins[1], maxs[1], 1000))

    # evaluate model on the dense grid
    Z = pred_func(np.c_[xs.flatten(), ys.flatten()])
    Z = Z.reshape(xs.shape)
    plt.figure(1)
    plt.subplot(121)
    # Plot the contour and training examples
    countour_plot = plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    countour_plot.collections[0].get_paths()[0].vertices

    plt.scatter(train_X[:, 0], train_X[:, 1], c = np.argmax(train_y,axis=1), s=50)
    plt.title('Peformance on training data')


    plt.subplot(122)
    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    plt.scatter(test_X[:, 0], test_X[:, 1], c=np.argmax(test_y, axis=1), s=50)
    plt.title('Peformance on testing data')
    plt.show()

def plot_forGuHan(X, y, clustering, indices, W, b):
    num_data = X.shape[0]
    num_clusters = len(W)

    #overall plot
    list_colors = ['green', 'blue', 'cyan', 'violet',
                   'magenta', 'black', 'purple', 'brown', 'orange','red' ]
    rvb = mpl.colors.ListedColormap(list_colors)
    cluster_colors = list()

    for i in range(num_data):
        cluster_colors.append(list_colors[clustering.predict(X)[i]])

    plt.figure(1)
    axes11 = plt.subplot(131)
    temp = np.reshape(y, (y.shape[0],))
    axes11.scatter(X[:, 0], X[:, 1], c=temp, s=50, cmap='viridis')
    axes11.set_title("Original lables")
    axes12 = plt.subplot(132)
    # plt.scatter(X[:, 0], X[:, 1], c=clustering.predict(X), s=50, cmap='viridis')
    temp = clustering.predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=temp, s=50, cmap=rvb,label=temp)
    # a = plt.scatter(X[:, 0], X[:, 1], s=50, color=cluster_colors)

    for (i, cla) in enumerate(set(temp)):
        xc = [p for (j, p) in enumerate(X[:,0]) if temp[j] == cla]
        yc = [p for (j, p) in enumerate(X[:,1]) if temp[j] == cla]
        cols = [c for (j, c) in enumerate(cluster_colors) if temp[j] == cla]
        axes12.scatter(xc, yc, c=cols, label=cla)
    axes12.legend(loc=4)
    axes12.set_title("Cluster formation")

    # plt.figure(2)
    axes13 = plt.subplot(133)
    # axes1 = plt.gca()
    #individual cluster plots
    # # the below should be uncommented only when plotting all clusters individually
    mins = np.amin(X, 0)
    mins = mins - 0.1 * np.abs(mins)
    maxs = np.amax(X, 0)
    maxs = maxs + 0.1 * maxs
    axes13.set_xlim([mins[0], maxs[0]])
    axes13.set_ylim([mins[1], maxs[1]])
    axes13.set_title("Clustered classification")

    error_vec = np.zeros([num_data,])
    for i in range(num_clusters):
        this_color = list_colors[i]
        this_w = 'w'+str(i+1)
        temp = indices['indices'+str(i+1)][0]
        color_map = np.sign(np.reshape(np.matmul(X[temp, :], W[this_w]) + b, (temp.shape[0],)))

        error_vec[temp] = np.abs(color_map- y[temp])
        cluster_accuracy = 1 - 0.5 * np.sum(error_vec[temp]) / len(temp)
        print("Accuracy for cluster:",str(i+1),"is:",cluster_accuracy)

        mins = np.amin(X[temp], 0)
        mins = mins - 0.1 * np.abs(mins)
        maxs = np.amax(X[temp], 0)
        maxs = maxs + 0.1 * maxs

        plt.figure(i+2)
        # plt.figure(1)
        # we add color_map and y[temp] below just to see the wrongly classified points
        axes13.scatter(X[temp, 0], X[temp, 1], c=color_map+y[temp], s=50, cmap='viridis')
        # x_vals = np.array(axes.get_xlim())
        x_vals = np.array([mins[0],maxs[0]])
        y_vals = -b/W[this_w][1] - W[this_w][0]/W[this_w][1] * x_vals
        axes13.plot(x_vals, y_vals,linewidth=2,color=this_color)

        # plt.figure(2)
        axes2 = plt.gca()
        plt.scatter(X[temp, 0], X[temp, 1], c=color_map + y[temp], s=50, cmap='viridis')
        x_vals = np.array([mins[0], maxs[0]])
        y_vals = -b / W[this_w][1] - W[this_w][0] / W[this_w][1] * x_vals
        plt.plot(x_vals, y_vals, linewidth=2, color=this_color)
        plt.text(x_vals[0] + 0.05, y_vals[0] + 0.05, str(i), color='red')
        axes2.add_patch(plt.Circle((x_vals[0] + 0.05, y_vals[0] + 0.05), radius=0.2, color='red', fill=False))

    overall_accuracy = 1-0.5*np.sum(error_vec)/num_data
    print("Overall accuracy is:", overall_accuracy)

# keep this line always here for plotting
plt.show(block=True)