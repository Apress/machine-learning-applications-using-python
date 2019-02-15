# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:43:13 2018

@author: PUNEETMATHUR
"""

import quandl
quandl.ApiConfig.api_key = 'INSERT YOU API KEY HERE'

# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call

data = quandl.get_table('WIKI/PRICES', ticker = ['AAPL', 'MSFT', 'WMT'], 
                        qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, 
                        date = { 'gte': '2015-12-31', 'lte': '2016-12-31' }, 
                        paginate=True)
data.head()

# create a new dataframe with 'date' column as index
new = data.set_index('date')

# use pandas pivot function to sort adj_close by tickers
clean_data = new.pivot(columns='ticker')

# check the head of the output
clean_data.head()

#Below script gets you Data from National Stock Exchange for a stock known as Oil India Limited
import quandl
quandl.ApiConfig.api_key = 'z1bxBq27SVanESKoLJwa'
quandl.ApiConfig.api_version = '2015-04-09'

import quandl
data = quandl.get('NSE/OIL')
data.head()
data.columns
data.shape

#Storing data in a flat file
data.to_csv("NSE_OIL.csv")

#A basic plot of the stocks data across the years
data['Close'].plot()
