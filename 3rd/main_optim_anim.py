import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import random

# original
from datafunctions import load_data
from main_basic_gp import calc_cov_K, Gaussian_process
from kernels import RBF_kernel_with_exp
from main_optim import optimize_gaussian_process, eval_func, gauss_grad
from animation import AnimDrawer

def make_data(X, Y):
    """
    Parameters
    ------------
    X : 
    Y : 
    Returns
    ----------
    X :
    Y : 
    """
    # parameters
    N = 1
    noise = 0.3
    min_x = -10. 
    max_x = 10.

    xs = [random.uniform(min_x, max_x) for _ in range(N)]
    ys = [np.sin(random_x) + random.gauss(0., noise) for random_x in xs]

    # concat with original data
    X = np.hstack((X, np.array(xs)))
    Y = np.hstack((Y, np.array(ys)))    

    return X, Y

def main():
    # initial data
    train_X = np.array([])
    train_Y = np.array([])

    # parameters
    iterations = 40
    results = {}

    for iteration in range(iterations):
        print("iteration = {}".format(iteration))
        # make data
        train_X, train_Y = make_data(train_X, train_Y)

        # initialize paramters
        TAU = np.log(1)
        SIGMA = np.log(1)
        ETA = np.log(1)

        param = {}
        param["tau"] = TAU
        param["sigma"] = SIGMA 
        param["eta"] = ETA

        init_param = [TAU, SIGMA, ETA]

        # optimize
        optimal_param = optimize_gaussian_process(train_X, train_Y, init_param, eval_func, gauss_grad)

        param = {}
        param["tau"] = optimal_param[0][0]
        param["sigma"] = optimal_param[0][1]
        param["eta"] = optimal_param[0][2]

        # predict
        x_min = -10.
        x_max = 10
        step = 350

        test_X = np.linspace(x_min, x_max, step)

        ave, var = Gaussian_process(test_X, train_X, train_Y, param, RBF_kernel_with_exp)
        # kernel
        kernel_x = np.linspace(-10., 10., 100)
        x_1 = [0.0]
        # index is same
        n_test = 0
        kernel_y = np.array([RBF_kernel_with_exp(x_1, x_2, n_test, n_test, param) for x_2 in kernel_x])   
        # save
        results["iteration = {} tau : {} sigma : {} eta : {}".format(iteration, round(param["tau"], 2), round(param["sigma"], 2), round(param["eta"], 2))] = [train_X, train_Y, test_X, ave, var, kernel_x, kernel_y]

    # anim
    animdrawer = AnimDrawer(results)
    animdrawer.draw_anim(interval=150)

if __name__ == "__main__":
    main()