import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle

# original
from animation import AnimDrawer
from datafunctions import load_data
from kernels import RBF_kernel
from main_basic_gp import calc_cov_K, calc_cov_k_s, Gaussian_process

def main():

    # load dataset
    train_X, train_Y = load_data("./data/ref_data.txt")
    
    x_min = -1.
    x_max = 3.5
    step = 100

    test_X = np.linspace(x_min, x_max, step)

    # parameters range
    mins = 0.1
    maxs = 5.0
    NUM = 500

    TAUs = [1.0] # np.linspace(mins, maxs, NUM) # theta_1
    SIGMAs = np.linspace(mins, maxs, NUM) # np.linspace(mins, maxs, NUM) # theta_2
    ETAs = [0.1] # np.linspace(mins, maxs, NUM) # theta_3

    results = OrderedDict()

    for TAU in TAUs:
        for SIGMA in SIGMAs:
            for ETA in ETAs:
                # paramters
                param = {}
                param["tau"] = TAU
                param["sigma"] = SIGMA 
                param["eta"] = ETA
                # predict
                ave, var = Gaussian_process(test_X, train_X, train_Y, param, RBF_kernel)
                # kernel
                kernel_x = np.linspace(-3., 3., 100)
                x_1 = [0.0]
                # index is same
                n_test = 0
                kernel_y = np.array([RBF_kernel(x_1, x_2, n_test, n_test, param) for x_2 in kernel_x])   
                # save
                results["tau = {}, sigma = {}, eta = {}".format(round(TAU, 2), round(SIGMA, 2), round(ETA, 2))] = [train_X, train_Y, test_X, ave, var,  kernel_x, kernel_y]

    # save
    with open("result.pkl", "wb") as f:
        pickle.dump(results, f)

    # anim
    animdrawer = AnimDrawer(results)
    animdrawer.draw_anim()

    """
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(train_X, train_Y, ".", c="b")
    
    axis.plot(test_X, ave_x, c="g")
    axis.fill_between(test_X, ave_x - 2. * np.sqrt(var_x), ave_x + 2. * np.sqrt(var_x), alpha=0.3, color="b")

    # kernel
    fig_2 = plt.figure()
    axis_2 = fig_2.add_subplot(111)
    
    test_X = np.linspace(-3., 3., 100)
    x_1 = [0.0]

    y = np.array([RBF_kernel(x_1, x_2, tau=TAU, sigma=SIGMA, eta=0.0) for x_2 in test_X])
    
    axis_2.plot(test_X, y)

    plt.show()
    """

if __name__ == "__main__":
    main()