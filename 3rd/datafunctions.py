import numpy as np

def load_data(path):
    """ load txt data
    Parameters
    ---------------
    path : str
        data path

    Returns
    --------
    X : numpy.ndarray, shape(N, D)
    Y : numpy.ndarray, shape(N)

    Example
    ----------
    >>> path = "./data/ref_data.txt"
    # ref_data.txt
    0.3 1.
    0.4 2.
    4.1 4.
    >>> X, Y = load_data(path)
    >>> X
    0.3
    0.4
    4.1
    >>> Y
    1.
    2.
    4.
    """
    data = []

    with open(path, mode="r") as txt_data:
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