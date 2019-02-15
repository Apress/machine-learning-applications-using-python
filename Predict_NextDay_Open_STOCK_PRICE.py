# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:30:01 2017

@author: PUNEETMATHUR
I am creating this script to predict next day Opening Price based on
Today's Closing Price for any given stock
"""

import numpy as np
import pandas as pd
import os

#Change your directory to wherever your dataset is stored
os.chdir("E:\\BUSINESS\\APRESS\\ApplicationsOfMachineLearning\\Chapter16\\")

#Loading the dataset of the company for which prediction is required
df=pd.read_csv("BalmerLawrieColtd.csv",parse_dates=['Date'])
print(df.head(1))
print(df.columns)
df.shape

#Selecting only relevant columns required for prediction
cols=['Date','Open','Close']
df=df[cols]

print(df.columns)
print(df.head(5))

# Checking data if Cleaning up data is required
df.isnull().any()
#df=df.dropna()
#df=df.replace("NA",0)
df.dtypes

#Sorting up data to plot historically ascending values in graph
df = df.sort_values(by='Date',ascending=True)

#Plotting the price of stock over the years
#What story does it tell?
import matplotlib.pyplot as plt
plt.plot(df['Date'],df['Close'])

#Now plot only for last one year and last 1 month 
df['Date'].dt.year==2017
mask=(df['Date'] > '2017-1-1') & (df['Date'] <= '2017-12-31')
print(df.loc[mask])
df2017=df.loc[mask]
print(df2017.head(5))
import matplotlib.pyplot as plt
plt.plot(df2017['Date'],df2017['Close'])

#Plotting last 1 month data on stock
mask=(df['Date'] > '2017-11-17') & (df['Date'] <= '2017-12-26')
print(df.loc[mask])
dfnovdec2017=df.loc[mask]
print(dfnovdec2017.head(5))
import matplotlib.pyplot as plt
plt.xticks( rotation='vertical')
plt.plot(dfnovdec2017['Date'],dfnovdec2017['Close'])

#Now calculating the Simple Moving Average of the Stock
#Simple Moving Average One Year
df2017['SMA'] = df2017['Close'].rolling(window=20).mean()
df2017.head(25)
df2017[['SMA','Close']].plot()


#Does the Open and Closing price of the stock follow very well?
df2017[['Open','Close']].plot()

#Checking the Correlation
df2017.corr()

#Simple Moving Average One Month
dfnovdec2017['SMA'] = dfnovdec2017['Close'].rolling(window=2).mean()
dfnovdec2017.head(25)
dfnovdec2017[['SMA','Close']].plot()

#Now creating NextDayOpen column for prediction
ln=len(df)
lnop=len(df['Open'])
print(lnop)
ii=0
df['NextDayOpen']=df['Open']
df['NextDayOpen']=0
for i in range(0,ln-1):
    print("Open Price: ",df['Open'][i])
    if i!=0:
        ii=i-1
    df['NextDayOpen'][ii]=df['Open'][i]
    print(df['NextDayOpen'][ii])

print(df['NextDayOpen'].head())

#Checking on the new data
print(df[['Open','NextDayOpen', 'Close']].head(5))

#Now checking if there is any correlation
dfnew=df[['Close','NextDayOpen']]
print(dfnew.head(5))
dfnew.corr()

#Now Creating the Prediction model as correlation is very high
#Importing the libraries
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


#Creating the features and target dataframes
price=dfnew['Close']
print(price)
print(dfnew.columns)
features=dfnew[['NextDayOpen']]
#Shuffling the data
price=shuffle(price, random_state=0)
features=shuffle(features,random_state=0)

#Dividing data into Train and Test
X_train, X_test, y_train, y_test= cross_validation.train_test_split(features,price,test_size=0.2, random_state=0)

#Linear Regression on Sensex data
reg= linear_model.LinearRegression()

X_train.shape
reg.fit(X_train, y_train)

y_pred= reg.predict(X_test)


print("Coefficients: ", reg.coef_)

#Mean squared error
print("mean squared error:  ",mean_squared_error(y_test,y_pred))

#Variance score
print("Variance score: ",   r2_score(y_test, y_pred))

#STANDARD DEVIATION
standarddev=price.std()

#Predict based on Opening BSE Sensex Index and Opening Volume
#In the predict function below enter the first parameter Open forNSE and 2nd Volume in Crores
sensexClosePredict=reg.predict([[269.05]])
#175 is the standard deviation of the Diff between Open and Close of sensex so this range
print("Stock Likely to Open at: ",sensexClosePredict , "(+-STD)")
print("Stock Open between: ",sensexClosePredict+standarddev , " & " , sensexClosePredict-standarddev)


