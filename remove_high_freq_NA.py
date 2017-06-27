"""
Acknowledgement: The code here has been written based on the content in the following link:
https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from read_data import read_data
from scipy.signal import correlate

def remove_high_freq_noise(df, sample_rate=1.0, order_list=range(1, 10),
                           cutoff_list=np.linspace(0.01, 0.1, 10),
                           err_type = "root-mean-sqrt"):
    """ Finds the best values for order and cutoff variables for a low-pass filter
        and then removes the high frequency noise.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input data
    sample_rate : float
        sample rate, Hz
    order_list : list
        Filter orders
    cutoff_list : list
        Filter cutoff values
    err_type : str
        Types of error that is used to find optimal window and win_type values.
        Valid inputs are root-mean-sqrt or mean-absolute


    Return
    ------
    pd.DataFrame
         Original input data together with the low-pass filtered data   
    """


#    dfn = pd.concat([df, df, df], ignore_index=True)
    dfn = df.copy()

    # find the best values for order and cutoff variables
    errs = np.full((len(order_list), len(cutoff_list)), 1e6)

    # loops through order_list and cutoff_list to find the best values
    for i, order in enumerate(order_list):
        for j, cutoff in enumerate(cutoff_list):
            filtered_data = butter_lowpass_filter(dfn.original,
                                                   cutoff, sample_rate, order)
            if err_type == "root-mean-sqrt": 
                errs[i, j] = np.sqrt(np.mean(np.square(dfn.true - filtered_data)))
            if err_type == "mean-absolute":
                errs[i, j] = np.mean(np.abs(dfn.true - filtered_data))

    # get the optimal order size and cutoff
    flat_index = np.argmin(errs)
    indx = np.unravel_index(flat_index, errs.shape)
    order = order_list[indx[0]]
    cutoff = cutoff_list[indx[1]]

    order = 3
    cutoff = 0.05
    #lag = np.argmax(correlate(filtered_data, df.true)) % dfn.shape[0]
    lag = np.argmax(correlate(df.true, filtered_data)) % dfn.shape[0]
    print "lag = ", lag
    filtered_data = np.roll(filtered_data, shift=lag)

    print "Best pair of order and cutoff are:"
    print "order =", order
    print "cutoff is ", cutoff

    # filter the data using the optimal order and cutoff 
    filtered_data = butter_lowpass_filter(dfn.original, cutoff, sample_rate, order)
    dfn['filtered'] = np.roll(filtered_data, shift=lag)

    # plot both the original and filtered signals
    fig2, ax2 = plt.subplots()
    dfn.plot(ax=ax2)
    ax2.grid()
    plt.show()

    # get the filter coefficients so we can check its frequency response
    b, a = butter_lowpass(cutoff, sample_rate, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    fig1, ax1 = plt.subplots()
    ax1.plot(0.5*sample_rate*w/np.pi, np.abs(h), 'b')
    ax1.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    ax1.axvline(cutoff, color='k')
    ax1.set_xlim(0, 0.5*sample_rate)
    ax1.set_title("Lowpass Filter Frequency Response")
    ax1.set_xlabel('Frequency [Hz]')
    ax1.grid()

def butter_lowpass(cutoff, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, sample_rate, order=5):
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    y = lfilter(b, a, data)
    return y

# run the code
if __name__ == "__main__":
    # read the data into a dataframe
    df = read_data("./original.mat")
    dfn = remove_high_freq_noise(df, sample_rate=1.0, order_list=range(1, 10),
                            cutoff_list=np.linspace(0.01, 0.1, 10))

