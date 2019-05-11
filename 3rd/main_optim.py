import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

# original
from datafunctions import load_data
from main_basic_gp import calc_cov_K, Gaussian_process
from kernels import RBF_kernel_with_exp

def kgauss_grad(x_i, x_j, d, n_i, n_j, param):
    """
    Parameters
    ---------------
    x_i : numpy.ndarray
    x_j : numpy.ndarray
    d : int
    n_i : int
        index of x_i
    n_j : int
        index of x_j
    param : dict
        parameters
    """
    eta = 0.

    if d == 0:
        if n_i == n_j:
            eta = np.exp(param["eta"])

        return np.exp(param["tau"]) * RBF_kernel_with_exp(x_i, x_j, n_i, n_j, param) - eta

    if d == 1:
        if n_i == n_j:
            eta = np.exp(param["eta"])

        return (RBF_kernel_with_exp(x_i, x_j, n_i, n_j, param) - eta) * (x_i - x_j) * (x_i - x_j) / np.exp(param["sigma"])

    if d == 2:
        return (np.exp(param["eta"]) if n_i == n_j else 0.)
    
    else:
        return 0.

def trace(A, B):
    """useful math func
    Parameters
    --------------
    A : numpy.ndarray, shape(N1, N2)
    B : numpy.ndarray, shape(N1, N2)
    Returns
    ---------
    val : float

    Examples
    ---------
    >>> A = [[1., 3.], [2., 5.]]
    >>> A = np.array(A)
    >>> B = [[0., 1.], [0., 2.]]
    >>> B = np.array(B)
    >>> trace(A,B)
    12.0
    """
    return np.sum(A*(B.T))

def gauss_grad(param_list, train_X, train_Y):
    """
    Parameters
    -----------
    param_list : list, shape(3)
    train_X : numpy.ndarray, shape(N, D)
    train_Y : numpy.ndarray, shape(N)

    Returns
    ---------
    grad : numpy.ndarray, shape(3)
    """
    N = train_X.shape[0] # number of data
    D = 3 # number of parameters
    param = {}
    param["tau"] = param_list[0]
    param["sigma"] = param_list[1] 
    param["eta"] = param_list[2]
    # get K
    K = calc_cov_K(train_X, param, RBF_kernel_with_exp)

    invK = np.linalg.inv(K)
    invKy = np.dot(invK, train_Y)

    grads = np.zeros(D)

    for d in range(D):
        # get grad
        G = np.zeros((N, N))
        for n_i, x_i in enumerate(train_X):
            for n_j, x_j in enumerate(train_X):
                kgrad = kgauss_grad(x_i, x_j, d, n_i, n_j, param) # make すべてのデータに対するgradient
                G[n_i, n_j] = kgrad

        assert (G == G.T).all(), "wrong val of G"
        grads[d] = trace(invK, G) - np.dot(invKy.T, np.dot(G, invKy)).flatten()[0]
    
    return grads

def eval_func(param_list, train_X, train_Y):
    """calc evidence of RBF kernel, - log |K_theta| - y^T K_theta^-1 y
    Parameters
    -----------
    param_list : list, shape(3)
    train_X : numpy.ndarray, shape(N, D)
    train_Y : numpy.ndarray, shape(N)

    Returns
    ----------
    val : float
    """
    param = {}
    param["tau"] = param_list[0]
    param["sigma"] = param_list[1] 
    param["eta"] = param_list[2]

    K = calc_cov_K(train_X, param, RBF_kernel_with_exp)
    invK = np.linalg.inv(K)
    detK = np.linalg.det(K)

    val = np.log(detK) + np.dot(train_Y.T, np.dot(invK, train_Y))

    return val

def optimize_gaussian_process(train_X, train_Y, init_param, eval_func, grad_func):
    """
    Parameters
    ------------
    train_X : numpy.ndarray, shape(N, D)
    train_Y : numpy.ndarray, shape(N)
    param : dict
        parameters of kernel
    eval_func : callable function
    grad_func : callable function
    
    Returns
    ---------
    optimal_param : dict
        parameters of kernel
    """
    # set init condition
    if train_X.ndim < 2:
        train_X = train_X[:, np.newaxis] # to 2 dim
    
    train_Y = train_Y[:, np.newaxis]
    
    # execute optimize
    optimal_param = fmin_l_bfgs_b(eval_func, init_param, fprime=grad_func, args=[train_X, train_Y])

    return optimal_param

def main():
    # load data
    train_X, train_Y = load_data("./data/ref_data.txt")

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

    # show
    print(optimal_param[0])

    param = {}
    param["tau"] = optimal_param[0][0]
    param["sigma"] = optimal_param[0][1]
    param["eta"] = optimal_param[0][2]

    print(param)

    # predict
    x_min = -1.
    x_max = 3.5
    step = 100

    test_X = np.linspace(x_min, x_max, step)

    ave_ys, var_ys = Gaussian_process(test_X, train_X, train_Y, param, RBF_kernel_with_exp)

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

    y = np.array([RBF_kernel_with_exp(x_1, x_2, n_test, n_train, param) for x_2 in test_X])
    
    axis_2.plot(test_X, y)

    plt.show()


if __name__ == "__main__":
    main()