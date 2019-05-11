import numpy as np
import matplotlib.pyplot as plt

from datafunctions import load_data
from kernels import RBF_kernel

def calc_cov_K(X, param, kernel_func):
    """
    Parameters
    -----------
    X : numpy.ndarray, shape(N, D)
        input
    param : dict
        valid parameters for kernel
    kernel_func : callable function

    Returns
    --------
    K : numpy.ndarray, shape(N, N)
    """
    # set init condition
    X = np.array(X)
    if X.ndim < 2:
        X = X[:, np.newaxis] # to 2 dim

    # get data num
    N = X.shape[0]

    # method 1
    K_test = np.zeros((N, N))

    for n_1, x_1 in enumerate(X):
        for n_2, x_2 in enumerate(X):    
            k = kernel_func(x_1, x_2, n_1, n_2, param)
            K_test[n_1, n_2] = k
            K_test[n_2, n_1] = k

    assert (K_test.T == K_test).all(), "not symmetric!! K =`{}".format(K_test)

    # print("test = \n{}".format(K_test))

    # method 2
    """
    # 1st get 内積の計算
    norm_X = np.sum(X**2, axis=1)

    # 2nd get 2つを掛けあわせた場合のx1*x2を算出
    multi_X = np.dot(X, X.T) # N * N

    assert multi_X.shape == (N, N), "size not correct multi_X = {}".format(multi_X)

    # ref : https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

    temp_K = np.tile(norm_X[:, np.newaxis], (1, N)) + np.tile(norm_X[:, np.newaxis].T, (N, 1)) - 2. * multi_X

    param["eta"] = 0.1
    K = param["tau"] * np.exp(- temp_K / (2. * param["sigma"] * param["sigma"])) + param["eta"] * np.eye(N)

    assert (K.T == K).all(), "not symmetric!! K =`{}".format(K)
    # print("test_2 = \n{}".format(K))
    assert (np.around(K, 5) == np.round(K_test, 5)).all(), "wrong"
    """

    return K_test

def calc_cov_k_s(test_x, train_X, param, kernel_func):
    """
    Parameters
    ----------
    test_x : numpy.ndarray, shape(D)
    train_X : numpy.ndarray, shape(N, D)
    param : dict
        valid parameters for kernel
    kernel_func : callable function
    """
    # initial condition
    assert test_x.shape[0] == train_X.shape[1], "size is wrong train_X = {}".format(train_X)

    # index is all different
    n_test = 0
    n_train = 1

    k_s = np.array([kernel_func(test_x, x, n_test, n_train, param) for x in train_X])

    assert train_X.shape[0] == k_s.shape[0], "size is wrong"

    return k_s[:, np.newaxis]

def Gaussian_process(test_X, train_X, train_Y, param, kernel_func):
    """
    Parameters
    ----------
    test_X : numpy.ndarray, shape(test_N, D)
    train_X : numpy.ndarray, shape(N, D)
    train_Y : numpy.ndarray, shape(N)
    param : dict
        parameters of kernel

    Returns
    -----------
    ave_ys : numpy.ndarray, shape(test_N)
    var_ys : numpy.ndarray, shape(test_N)
    """
    # set init condition
    if train_X.ndim < 2:
        train_X = train_X[:, np.newaxis] # to 2 dim
    
    if test_X.ndim < 2:
        test_X = test_X[:, np.newaxis] # to 2 dim
    
    # calc kernel mat
    K = calc_cov_K(train_X, param, kernel_func)

    # inv
    invK = np.linalg.inv(K)

    ave_ys = []
    var_ys = []

    # predict for each input x
    for test_x in test_X:
        # self
        n_test = 0 # same index
        k_ss = kernel_func(test_x, test_x, n_test, n_test, param)
        # other
        k_s = calc_cov_k_s(test_x, train_X, param, kernel_func)

        ave = np.dot(np.dot(k_s.T, invK), train_Y[:, np.newaxis]) 
        var = k_ss - np.dot(np.dot(k_s.T, invK), k_s)

        # save
        ave_ys.append(ave)
        var_ys.append(var)

    return np.array(ave_ys).flatten(), np.array(var_ys).flatten()

def main():

    # load dataset
    train_X, train_Y = load_data("./data/ref_data.txt")
    
    x_min = -1.
    x_max = 3.5
    step = 100

    test_X = np.linspace(x_min, x_max, step)

    # paramters
    TAU = 1.0 # theta_1 
    SIGMA = 5.0 # theta_2
    ETA = 0.1 # theta_3

    param = {}
    param["tau"] = TAU
    param["sigma"] = SIGMA 
    param["eta"] = ETA

    # predict
    ave_ys, var_ys = Gaussian_process(test_X, train_X, train_Y, param, RBF_kernel)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(train_X, train_Y, ".", c="b")
    
    axis.plot(test_X, ave_ys, c="g")
    axis.fill_between(test_X, ave_ys - 2. * np.sqrt(var_ys), ave_ys + 2. * np.sqrt(var_ys), alpha=0.3, color="b")

    # kernel
    fig_2 = plt.figure()
    axis_2 = fig_2.add_subplot(111)
    
    test_X = np.linspace(-3., 3., 100)
    x_1 = [0.0]

    # index is all different
    n_test = 0
    n_train = 1

    y = np.array([RBF_kernel(x_1, x_2, n_test, n_train, param) for x_2 in test_X])
    
    axis_2.plot(test_X, y)

    plt.show()

if __name__ == "__main__":
    main()