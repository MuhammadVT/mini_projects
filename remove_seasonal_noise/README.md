This mini-project implements a few methods to remove seasonal noise from a small size of data set.

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)

You will also need to install [Jupyter Notebook](http://ipython.org/notebook.html) to run the demo.ipynb notebook 

### Run

In a terminal or command window, navigate to the project directory `remove_seasonal_noise/`
and run the following command:

```bash
jupyter notebook demo.ipynb
```

This will open the Jupyter Notebook software and a demo file in your browser.


### Data

The data analyzed here is a small set of data that contains 360 number of points. The data is in .mat files, `True.mat` and `original.mat`. 
`original.mat` has the `original` data which is obtained from an observation, and `True.mat` file contains the simulation data labeled as `true`.
The simulation data is there to tell us what the actual data might look like. 

**Features**
1.  `original`: floating piont data from an observation
2. `true`: data from a simulation
