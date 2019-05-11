#!/usr/local/bin/python
#
#    gpr.py -- Gaussian process regression.
#    $Id: gpr.py,v 1.3 2017/11/12 04:51:52 daichi Exp $
#
import sys
import numpy as np
from pylab import *
from numpy import exp,sqrt
from numpy.linalg import inv

# plot parameters
N    = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3

# GP kernel parameters
eta   = 0.1
tau   = 1
sigma = 5.0

def kgauss (params):
    [tau,sigma] = params
    return lambda x,y: tau * exp (-(x - y)**2 / (2 * sigma * sigma))

def test_gauss(x, y):
    print(exp(-(x - y)**2))
    return tau * exp (-(x - y)**2 / (2 * sigma * sigma))

def kv (x, xtrain, kernel):
    print("x = {}".format(x))
    print("x_train = {}".format(xtrain))
    
    print(np.array ([test_gauss(x,xi) for xi in xtrain]))

    return np.array ([kernel(x,xi) for xi in xtrain])

def kernel_matrix (xx, kernel):
    N = len(xx)
    return np.array (
        [kernel (xi, xj) for xi in xx for xj in xx]
    ).reshape(N,N) + eta * np.eye(N)

def gpr (xx, xtrain, ytrain, kernel):
    K = kernel_matrix (xtrain, kernel)
    print("K = {} ".format(K))
    Kinv = inv(K)
    ypr = []; spr = []
    for x in xx:
        s = kernel (x,x) + eta
        print("s = {} ".format(s))
        k = kv (x, xtrain, kernel)
        print("k = {} ".format(k))
        print(x)
        # input()
        ypr.append (k.T.dot(Kinv).dot(ytrain))
        spr.append (s - k.T.dot(Kinv).dot(k))
    return ypr, spr

def gpplot (xx, xtrain, ytrain, kernel, params):
    ypr,spr = gpr (xx, xtrain, ytrain, kernel(params))
    plot (xtrain, ytrain, 'bx', markersize=16)
    plot (xx, ypr, 'b-')
    fill_between (xx, ypr - 2*sqrt(spr), ypr + 2*sqrt(spr), color='#ccccff')

    plt.show()

def usage ():
    print ('usage: gpr.py train output')
    print ('$Id: gpr.py,v 1.3 2017/11/12 04:51:52 daichi Exp $')
    sys.exit (0)

def main ():
    if len(sys.argv) < 2:
        usage ()
    else:
        train = np.loadtxt (sys.argv[1], dtype=float)
        
    xtrain = train.T[0]
    ytrain = train.T[1]
    kernel = kgauss
    params = [tau,sigma]
    xx     = np.linspace (xmin, xmax, N)

    gpplot (xx, xtrain, ytrain, kernel, params)

if __name__ == "__main__":
    main ()