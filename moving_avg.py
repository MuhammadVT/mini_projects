import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from read_data import read_data

def moving_average(df, window=10, optimal_window=True, iter_num=30):
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

    Return
    ------
    pandas.DataFrame
        Original input data together with the moving_average


    """

    dfn = df.copy()

    # find the optimal window size between 2 and iter_num+1
    if optimal_window:
        err_rms = []
        err_abs = []
        windows = range(2, iter_num+2)

        # loops through windows to find the best window value
        for i in windows:
            mean_data = dfn.original.rolling(i, min_periods=1, center=True).mean()
            err_rms.append(np.sqrt(np.mean(np.square(dfn.true - mean_data))))
            err_abs.append(np.mean(np.abs(dfn.true - mean_data)))

        # get the optimal window size
        #window = windows[np.argmin(err_rms)]
        window = windows[np.argmin(err_abs)]

        # calculate the moving average that corresponds to the optimal window 
        dfn['moving_avg'] = dfn.original.rolling(window, min_periods=1, center=True).mean()

    else:
        #dfn['mean'] = pd.rolling_mean(dfn.original, window, center=True)
        dfn['moving_avg'] = dfn.original.rolling(window, min_periods=1, center=True).mean()
    dfn.plot()
    plt.show()

# run the code
optimal_window = True
iter_num = 30

# read the data
df = read_data("./original.mat")

moving_average(df, window=10, optimal_window=optimal_window)

