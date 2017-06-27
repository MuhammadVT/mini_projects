import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from read_data import read_data

def moving_average(df, window=10, optimal_window=True, iter_num=30,
                   err_type = "root-mean-sqrt"):
    """Calculates the moving average to remove noise

    Parameters
    ----------
    df : pandas.DataFrame
        The input data
    window : int
        window size
    optimal_window : bool
        If set to True then looks for the best optimal window based mean-absolute error
    iter_num : int
        Number of iteration to find the best window size withing range(2, iter_num+2).
        Only works if optimal_window is set to True.
    err_type : str
        Types of error that is used to find optimal window and win_type values.
        Valid inputs are root-mean-sqrt or mean-absolute

    Return
    ------
    pandas.DataFrame
        Original input data together with the moving_average

    """

    dfn = df.copy()

    # find the optimal window size between 2 and iter_num+1
    if optimal_window:
        win_types = [ 
            "boxcar", 
            "triang", 
            "blackman", 
            "hamming", 
            "bartlett", 
            "parzen", 
            "bohman", 
            "blackmanharris", 
            "nuttall", 
            "barthann"] 
#            "kaiser",
#            "gaussian",
#            "general_gaussian"
#            "slepian"]
        windows = range(2, iter_num+2)
        errs = np.full((len(windows), len(win_types)), 1e6)

        # loops through windows to find the best window win_type values
        for i, window in enumerate(windows):
            for j, win_type in enumerate(win_types):
                mean_data = dfn.original.rolling(window, min_periods=1,
                                                 win_type=win_types[j],
                                                 center=True).mean()
                if err_type == "root-mean-sqrt": 
                    errs[i, j] = np.sqrt(np.mean(np.square(dfn.true - mean_data)))
                if err_type == "mean-absolute":
                    errs[i, j] = np.mean(np.abs(dfn.true - mean_data))

        # get the optimal window size and win_type
        flat_index = np.argmin(errs)
        indx = np.unravel_index(flat_index, errs.shape)
        window = windows[indx[0]]
        win_type = win_types[indx[1]]
        print "Best pair of window and win_type are:"
        print "window =", window
        print "win_type is ", win_type

        # calculate the moving average that corresponds to the optimal window 
        dfn['moving_avg'] = dfn.original.rolling(window, min_periods=1,
                                                 win_type=win_type,
                                                 center=True).mean()
    else:
        dfn['moving_avg'] = dfn.original.rolling(window, min_periods=1,
                                                 center=True).mean()

    # plot the data
    dfn.plot()
    plt.show()

    return dfn

# run the code
optimal_window = True
#optimal_window = False
iter_num = 50
err_type = "root-mean-sqrt"
#err_type = "mean-absolute"

# read the data
df = read_data("./original.mat")
dfn = moving_average(df, window=10, optimal_window=optimal_window, err_type=err_type)

