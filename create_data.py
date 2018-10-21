import numpy as np
import random

def create_LSD(N=1000,m=-2,c=1,x1ll=-2,x1rl=2,x2ll=-2,x2rl=4):
    #creates linearly separable dataset (one line)
    delta = 0.025;
    xx = np.arange(x1ll, x1rl, delta)
    data = np.empty([N,3]) #X matrix
    data[:,0] = np.random.uniform(x1ll,x1rl,size=(N,))
    data[:,1] = np.random.uniform(x2ll,x2rl,size=(N,))
    for i in range(N):
        if data[i,1]-m*data[i,0]-c>0:
            data[i,2] = 1
            # data[i,2] = random.sample([0,1],1)[0]
        else:
            data[i,2] = 0
            # data[i,2] = random.sample([0,1],1)[0]
    data_file = "2D_LSD.csv"
    np.savetxt(data_file, data, delimiter=",")
    data_params = np.array([N,m,c,x1ll,x1rl,x2ll,x2rl])
    return data_file, data_params

def create_PLSD(N=1000,m1=-2,c1=1,m2=2,c2=-1,x1ll=-2,x1rl=2,x2ll=-2,x2rl=4):
    #creates linearly separable dataset (two lines)
    delta = 0.025;
    xx = np.arange(x1ll, x1rl, delta)
    data = np.empty([N,3]) #X matrix
    data[:,0] = np.random.uniform(x1ll,x1rl,size=(N,))
    data[:,1] = np.random.uniform(x2ll,x2rl,size=(N,))
    for i in range(N):
        if (data[i,1]-m1*data[i,0]-c1>0) & (data[i,1]-m2*data[i,0]-c2>0):
            data[i,2] = 1
        elif (data[i,1]-m1*data[i,0]-c1>0) & (data[i,1]-m2*data[i,0]-c2<0):
            data[i,2] = 2
        elif (data[i,1]-m1*data[i,0]-c1<0) & (data[i,1]-m2*data[i,0]-c2>0):
            data[i,2] = 3
        else:
            data[i,2] = 0 #one of the labels HAS to be 0 otherwise our_MLP3 throws error
    data_file = "2D_PLSD.csv"
    np.savetxt(data_file, data, delimiter=",")
    data_params = np.array([N,m1,c1,m2,c2,x1ll,x1rl,x2ll,x2rl])
    return data_file, data_params

def create_PPLSD(N=1000,m1=-2,c1=1,m2=2,c2=-1,m3=-1/16,c3=0.5,x1ll=-2,x1rl=2,x2ll=-2,x2rl=4):
    #creates linearly separable dataset (three lines)
    delta = 0.025;
    xx = np.arange(x1ll, x1rl, delta)
    data = np.empty([N,3]) #X matrix
    data[:,0] = np.random.uniform(x1ll,x1rl,size=(N,))
    data[:,1] = np.random.uniform(x2ll,x2rl,size=(N,))
    for i in range(N):
        if (data[i,1]-m1*data[i,0]-c1>0) & (data[i,1]-m2*data[i,0]-c2>0) & (data[i,1]-m3*data[i,0]-c3>0):
            data[i,2] = 1
        elif (data[i,1]-m1*data[i,0]-c1>0) & (data[i,1]-m2*data[i,0]-c2>0) & (data[i,1]-m3*data[i,0]-c3<0):
            data[i,2] = 2
        elif (data[i,1]-m1*data[i,0]-c1>0) & (data[i,1]-m2*data[i,0]-c2<0) & (data[i,1]-m3*data[i,0]-c3>0):
            data[i,2] = 3
        elif (data[i,1]-m1*data[i,0]-c1<0) & (data[i,1]-m2*data[i,0]-c2>0) & (data[i,1]-m3*data[i,0]-c3>0):
            data[i,2] = 4
        elif (data[i,1]-m1*data[i,0]-c1>0) & (data[i,1]-m2*data[i,0]-c2<0) & (data[i,1]-m3*data[i,0]-c3<0):
            data[i,2] = 5
        elif (data[i,1]-m1*data[i,0]-c1<0) & (data[i,1]-m2*data[i,0]-c2>0) & (data[i,1]-m3*data[i,0]-c3<0):
            data[i,2] = 6
        elif (data[i,1]-m1*data[i,0]-c1<0) & (data[i,1]-m2*data[i,0]-c2<0) & (data[i,1]-m3*data[i,0]-c3>0):
            data[i,2] = 7
        else:
            data[i,2] = 0 #one of the labels HAS to be 0 otherwise our_MLP3 throws error
    data_file = "2D_PPLSD.csv"
    np.savetxt(data_file, data, delimiter=",")
    data_params = np.array([N,m1,c1,m2,c2,m3,c3,x1ll,x1rl,x2ll,x2rl])
    return data_file, data_params

def create_PolySD(N=1000,coeff=[2,3,1,1,1],x1ll=-4,x1rl=4,x2ll=-1,x2rl=70):
    def eval_poly(coeff,xx):
        degree = len(coeff) - 1
        yy = 0
        for i in range(len(coeff)):
            yy = yy+coeff[i]*np.power(xx,(degree-i))
        return yy

    #creates polynomially separable dataset
    delta = 0.025
    data = np.empty([N,3]) #X matrix
    data[:,0] = np.random.uniform(x1ll,x1rl,size=(N,))
    data[:,1] = np.random.uniform(x2ll,x2rl,size=(N,))
    for i in range(N):
        if eval_poly(coeff,data[i,0])-data[i,1]>0:
            data[i,2] = 1
        else:
            data[i,2] = 0
    data_file = "2D_PolySD.csv"
    np.savetxt(data_file, data, delimiter=",")
    data_params = np.concatenate((np.array([N]), coeff, np.array([x1ll, x1rl, x2ll, x2rl])))
    return data_file, data_params

def process_PolySD(our_coeff):
    # generating the appropriate arguments for the polynomial such that the range of x lies between leftmost
    # and rightmost root of polynomial and y range is between the values of polynomial at these two points
    # this doesn't give nice ranges so do explicitly
    our_roots = np.roots(our_coeff)
    our_min_x = min(np.real(our_roots))
    our_max_x = max(np.real(our_roots))
    our_range_x = our_max_x - our_min_x
    our_linspace_x = np.linspace(our_min_x, our_max_x, our_range_x / 0.1)
    our_min_x = our_min_x - 0.1 * our_range_x
    our_max_x = our_max_x + 0.1 * our_range_x
    temp = np.polyval(our_coeff, our_linspace_x)
    our_min_y = min(temp)
    our_max_y = max(temp)
    our_range_y = our_max_y - our_min_y
    our_min_y = our_min_y - 0.1 * our_range_y
    our_max_y = our_max_y + 0.1 * our_range_y
    return our_min_x, our_max_x, our_min_y, our_max_y


