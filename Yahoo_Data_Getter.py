# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:32:32 2017

@author: PUNEETMATHUR
"""

import numpy as np
import pandas as pd
#import pandas.io.data as web
from pandas_datareader import data, wb

sp500= data.DataReader('^GSPC', data_source='yahoo', start='1/1/2000', end='1/12/2017')
#sp500= data.DataReader('^GSPC', data_source='yahoo')
sp500.ix['2010-01-04']
sp500.info()
print(sp500)
print(sp500.columns)
print(sp500.shape)

import matplotlib.pyplot as plt
plt.plot(sp500['Close'])

# now calculating the 42nd Days and 252 days trend for the index
sp500['42d']= np.round(pd.rolling_mean(sp500['Close'], window=42),2)
sp500['252d']= np.round(pd.rolling_mean(sp500['Close'], window=252),2)

#Look at the data
sp500[['Close','42d','252d']].tail()
plt.plot(sp500[['Close','42d','252d']])

