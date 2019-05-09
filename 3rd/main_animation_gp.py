import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle

# original
from animation import AnimDrawer

def load_data():
    """
    Parameters
    ---------------

    Returns
    --------

    """
    data = []

    with open("./data/ref_data.txt", mode="r") as txt_data:
        raw_data = txt_data.readlines()

        for raw_data_row in raw_data:
            raw_data_row = raw_data_row.replace("\n", "")
            raw_data_row = raw_data_row.split(" ")

            data_row = list(map(float, raw_data_row)) # to float
            data.append(data_row)

    data = np.array(data)

    """
    data_fig = plt.figure()
    axis = data_fig.add_subplot(111)
    axis.plot(data[:, 0], data[:, 1], ".", c="b")
    plt.show()
    """

    return data[:, 0], data[:, 1] 

def RBF_kernel(x_1, x_2, eta=0.1, tau=1., sigma=1.):
    """
    Parameters
    -----------
    x_1 : numpy.ndarray
    x_2 : numpy.ndarray
    """
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    k = tau * np.exp(-np.sum((x_1 - x_2)**2) / (2. * sigma * sigma)) + eta # ここの書き方はいろいろありそう
    # k = tau * np.exp(-(x_1 - x_2)**2 / (2. * sigma * sigma)) + eta # ここの書き方はいろいろありそう

    return k

def calc_cov_K(X, TAU, SIGMA, ETA):
    """
    Parameters
    -----------
    X : numpy.ndarray, shape(N, D)

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

    for n1, x1 in enumerate(X):
        for n2, x2 in enumerate(X):
            ETA = 0.0
            if n1 == n2:
                ETA = 0.1

            k = RBF_kernel(x1, x2, eta=ETA, tau=TAU, sigma=SIGMA)
            K_test[n1, n2] = k
            K_test[n2, n1] = k

    assert (K_test.T == K_test).all(), "not symmetric!! K =`{}".format(K_test)

    # print("test = \n{}".format(K_test))

    # method 2
    # 1st get 内積の計算
    norm_X = np.sum(X**2, axis=1)

    # 2nd get 2つを掛けあわせた場合のx1*x2を算出
    multi_X = np.dot(X, X.T) # N * N

    assert multi_X.shape == (N, N), "size not correct multi_X = {}".format(multi_X)

    # ref : https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

    temp_K = np.tile(norm_X[:, np.newaxis], (1, N)) + np.tile(norm_X[:, np.newaxis].T, (N, 1)) - 2. * multi_X

    K = TAU * np.exp(- temp_K / (2. * SIGMA * SIGMA)) + ETA * np.eye(N)

    assert (K.T == K).all(), "not symmetric!! K =`{}".format(K)

    # print("test_2 = \n{}".format(K))
    assert (np.around(K, 5) == np.round(K_test, 5)).all(), "wrong"

    return K

def calc_cov_k_s(test_x, train_X, TAU, SIGMA):
    """
    Parameters
    ----------
    test_x : numpy.ndarray, shape(D)
    train_X : numpy.ndarray, shape(N, D)
    """
    # initial condition
    assert test_x.shape[0] == train_X.shape[1], "size is wrong train_X = {}".format(train_X)

    k_s = np.array([RBF_kernel(test_x, x, eta=0.0, sigma=SIGMA, tau=TAU) for x in train_X])

    assert train_X.shape[0] == k_s.shape[0], "size is wrong"

    return k_s[:, np.newaxis]

def Gaussian_process(test_X, train_X, train_Y, TAU, SIGMA, ETA, kernel="RBF"):
    """
    Parameters
    ----------
    test_X : numpy.ndarray, shape(test_N, D)
    train_X : numpy.ndarray, shape(N, D)
    train_Y : numpy.ndarray, shape(N)

    Returns
    -----------
    ave_y : 
    var_y : 
    """
    # set init condition
    if train_X.ndim < 2:
        train_X = train_X[:, np.newaxis] # to 2 dim
    
    if test_X.ndim < 2:
        test_X = test_X[:, np.newaxis] # to 2 dim
    
    # calc kernel mat
    K = calc_cov_K(train_X, TAU, SIGMA, ETA)

    # inv
    invK = np.linalg.inv(K)

    ave = []
    var = []

    # predict for each input x
    for test_x in test_X:
        # self
        k_ss = RBF_kernel(test_x, test_x, TAU, SIGMA, ETA)
        # other
        k_s = calc_cov_k_s(test_x, train_X, TAU, SIGMA)

        ave_y = np.dot(np.dot(k_s.T, invK), train_Y[:, np.newaxis]) 
        var_y = k_ss - np.dot(np.dot(k_s.T, invK), k_s)

        # save
        ave.append(ave_y)
        var.append(var_y)

    return np.array(ave).flatten(), np.array(var).flatten()

def main():

    # load dataset
    train_X, train_Y = load_data()
    
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
                # predict
                ave, var = Gaussian_process(test_X, train_X, train_Y, TAU, SIGMA, ETA)
                # kernel
                kernel_x = np.linspace(-3., 3., 100)
                x_1 = [0.0]
                kernel_y = np.array([RBF_kernel(x_1, x_2, tau=TAU, sigma=SIGMA, eta=0.0) for x_2 in kernel_x])   
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