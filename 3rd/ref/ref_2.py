#!/usr/local/bin/python
#
#    gpr.py -- Gaussian process regression.
#    $Id: gpr.py,v 1.14 2018/03/09 00:55:08 daichi Exp $
#
#
import sys
import numpy as np
# from pylab import *
import matplotlib.pyplot as plt
from numpy.linalg import det,inv
from scipy.optimize import minimize, fmin_l_bfgs_b

# plot parameters
N    = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3
blue = '#ccccff'

def kgauss (params):
    [tau,sigma,eta] = params
    # print(params)
    return lambda x,y,train=True: \
        np.exp(tau) * np.exp (-(x - y)**2 / np.exp(sigma)) + \
        (np.exp(eta) if (train and x == y) else 0)

def kgauss_grad (xi,xj,d,kernel,params):
    if d == 0:
        return np.exp(params[d]) * kernel(params)(xi, xj)
    if d == 1:
        return kernel(params)(xi, xj) * \
               (xi - xj) * (xi - xj) / np.exp(params[d])
    if d == 2:
        return (np.exp(params[d]) if xi == xj else 0)
    else:
        return 0
    
def kv (x, xtrain, kernel):
    return np.array ([kernel(x,xi,False) for xi in xtrain])

def kernel_matrix (xx, kernel):
    N = len(xx)
    return np.array (
        [kernel (xi, xj) for xi in xx for xj in xx]
    ).reshape(N,N)

def gpr (xx, xtrain, ytrain, kernel):
    K = kernel_matrix (xtrain, kernel)
    Kinv = inv(K)
    ypr = []; spr = []
    for x in xx:
        s = kernel (x,x)
        k = kv (x, xtrain, kernel)
        ypr.append (k.T.dot(Kinv).dot(ytrain))
        spr.append (s - k.T.dot(Kinv).dot(k))
    return ypr, spr

def tr(A,B):
    return (A*B.T).sum()

def printparam (params):
    print (params)

def loglik (params,xtrain,ytrain,kernel,kgrad):
    K = kernel_matrix (xtrain, kernel(params))
    Kinv = inv(K)
    return np.log(det(K)) + ytrain.T.dot(Kinv).dot(ytrain)
    # return (N * log(2*np.pi) + \
    #         log(det(K)) + ytrain.T.dot(Kinv).dot(ytrain)) / 2

def gradient (params,xtrain,ytrain,kernel,kgrad):
    K = kernel_matrix (xtrain, kernel(params))
    # print("grad K = {}".format(K))
    Kinv = inv(K)
    Kinvy = Kinv.dot(ytrain)
    D = len(params)
    N = len(xtrain)
    grad = np.zeros(D)
    for d in range(D):
        G = np.array (
                [kgrad (xi, xj, d, kernel, params)
                for xi in xtrain for xj in xtrain]
            ).reshape(N,N)
        
        print("G = {}".format(G))
        grad[d] = tr(Kinv,G) - Kinvy.dot(G).dot(Kinvy)
        print("grad[d] = {}".format(grad[d]))
        # input()

    print("grad = {}".format(grad))
    # input()
    return grad

def numgrad (params,xtrain,ytrain,kernel,kgrad,eps=1e-6):
    D = len(params)
    ngrad = np.zeros (D)
    for d in range(D):
        lik = loglik (params,xtrain,ytrain,kernel,kgrad)
        params[d] += eps
        newlik = loglik (params,xtrain,ytrain,kernel,kgrad)
        params[d] -= eps
        ngrad[d] = (newlik - lik) / eps
    return ngrad

def optimize (xtrain, ytrain, kernel, kgrad, init):
    res = minimize (loglik, init, args = (xtrain,ytrain,kernel,kgrad),
                    jac = gradient, # numgrad
                    method = 'BFGS', callback = printparam,
                    options = {'gtol' : 1e-4, 'disp' : True})
    # print res.message
    return res.x

def optimize1 (xtrain, ytrain, kernel, kgrad, init):
    x,flog,feval,status = SCG (loglik, gradient, init,
                               optargs=[xtrain,ytrain,kernel,kgrad])
    # print status
    return x

def optimize2 (xtrain, ytrain, kernel, kgrad, init):
    x,f,d = fmin_l_bfgs_b (loglik, init, fprime=gradient,
                           args=[xtrain,ytrain,kernel,kgrad],
                           iprint=0, maxiter=1000)
    # print d
    return x

def gpplot (xtrain, ytrain, kernel, params):
    xx = np.linspace (xmin, xmax, N)
    ypr,spr = gpr (xx, xtrain, ytrain, kernel(params))
    plt.plot (xtrain, ytrain, 'bx', markersize=16)
    plt.plot (xx, ypr, 'b-')
    plt.fill_between (xx, ypr - 2*np.sqrt(spr), ypr + 2*np.sqrt(spr), color=blue)
    plt.show()

def usage ():
    # print 'usage: gpr.py train [output]'
    # print '$Id: gpr.py,v 1.14 2018/03/09 00:55:08 daichi Exp $'
    sys.exit (0)

def main ():
    if len(sys.argv) < 2:
        usage ()
    else:
        train = np.loadtxt (sys.argv[1], dtype=float)
        # kernel parameters
        tau   = np.log(1)
        sigma = np.log(1)
        eta   = np.log(1)
        
    xtrain = train.T[0]
    ytrain = train.T[1]
    kernel = kgauss
    kgrad  = kgauss_grad
    params = np.array ([tau, sigma, eta])

    # print 'grad  =', gradient (params, xtrain, ytrain, kernel, kgrad)
    # print 'ngrad =', numgrad (params, xtrain, ytrain, kernel, kgrad)

    params = optimize2 (xtrain, ytrain, kernel, kgrad, params)
    print(params)
    # print 'params =',; print params
    gpplot (xtrain, ytrain, kernel, params)

if __name__ == "__main__":
    main ()