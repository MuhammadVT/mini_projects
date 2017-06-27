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

def remove_high_freq_noise(df, sample_rate=1.0, order=3, cutoff=0.1):
    """ Finds the best values for order and cutoff variables for a low-pass filter
        and then removes the high frequency noise.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input data
    sample_rate : float
        sample rate, Hz
    order : int
        Filter order
    cutoff : float
        Filter cutoff value

    Return
    ------
    pd.DataFrame
         Original input data together with the low-pass filtered data   
    """

    dfn = df.copy()

    # filter the data
    filtered_data = butter_lowpass_filter(dfn.original, cutoff, sample_rate, order=order)

    # offset the lag time in the filtered signal
    lag = np.argmax(correlate(df.true, filtered_data)) % dfn.shape[0]
    filtered_data = np.roll(filtered_data, shift=lag)
    dfn['filtered'] = filtered_data

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

    return dfn

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
    dfn = remove_high_freq_noise(df, sample_rate=1.0, order=3,
                            cutoff=0.05)

    # plot both the original and filtered signals
    fig2, ax2 = plt.subplots()
    dfn.plot(ax=ax2)
    ax2.grid()
    plt.show()


