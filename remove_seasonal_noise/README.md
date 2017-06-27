This mini-project shows a few approaches to remove seasonal noise from a small size of data set

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

### Run

In a terminal or command window, navigate to the project directory `remove_seasonal_noise/`
and run one of the following commands:

```bash
jupyter notebook demo.ipynb
```

This will open the Jupyter Notebook software and a demo file in your browser.


### Data

The data analyzed here is a small set of data that contains 360 number of points. The data is in .mat file. 
It has two columns, first column is the original data which is obtained from an observation, and the second colums is a simulation data labeled as "true".
The simulation data is there to tell us what the actual data might look like. 

**Features**
1.  `original`: floating piont data from observation
2. `true`: data from a simulation
