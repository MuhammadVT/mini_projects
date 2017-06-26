from scipy import io
import pandas as pd

def read_data(fname):
    """ reads data from a .m file 
    Parameters
    ----------
    fname : str
        file path

    Return
    ------
    padas.DataFrame
    """
    # load the data from .m file
    dat_o = io.loadmat("./original.mat")
    dat_o = dat_o["Original"][0]
    dat_t = io.loadmat("./True.mat")
    dat_t = dat_t["True"][0]

    # create a dataframe
    df = pd.DataFrame(data=zip(dat_o, dat_t), columns=['original', 'true'])

    return 
    df

