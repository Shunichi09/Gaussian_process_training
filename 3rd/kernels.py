import numpy as np

def RBF_kernel(x_1, x_2, n_1, n_2, param):
    """ calculate RBF kernel
    Parameters
    -----------
    x_1 : numpy.ndarray or float
    x_2 : numpy.ndarray or float
    n_1 : int
        index of x_1
    n_2 : int
        index of x_2
    param : dict
    
    Returns
    --------------
    k : float
    """
    eta = param["eta"]
    tau = param["tau"]
    sigma = param["sigma"]

    if n_1 != n_2: # check index
        # print("x_1, x_2 is not equal!!")
        eta = 0.
    
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    k = tau * np.exp(-np.sum((x_1 - x_2)**2) / (2. * sigma * sigma)) + eta # ここの書き方はいろいろありそう
    # k = tau * np.exp(-(x_1 - x_2)**2 / (2. * sigma * sigma)) + eta # ここの書き方はいろいろありそう

    return k

def RBF_kernel_with_exp(x_1, x_2, n_1, n_2, param):
    """ calculate RBF kernel
    Parameters
    -----------
    x_1 : numpy.ndarray or float
    x_2 : numpy.ndarray or float
    param : dict
    
    Returns
    --------------
    k : float
    """
    eta = np.exp(param["eta"])
    tau = np.exp(param["tau"])
    sigma = np.exp(param["sigma"])
    
    if n_1 != n_2: # check index
        # print("x_1, x_2 is not equal!!")
        eta = 0.

    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    k = tau * np.exp(-np.sum((x_1 - x_2)**2) / (sigma)) + eta # ここの書き方はいろいろありそう

    return k