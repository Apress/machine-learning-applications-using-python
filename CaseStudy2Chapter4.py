# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:16:52 2018

@author: PMAUTHOR
"""

import pandas as pd
from io import StringIO
import requests
import os
os.getcwd()

fname="Food_Raw_Data.csv"
hospitals= pd.read_csv(fname, low_memory=False, index_col=False)
df= pd.DataFrame(hospitals)
print(df.head(1))

print(df.size)
print(df.shape)
print(df.columns)
df.dtypes

#Check if there are any columns with empty/null dataset
df.isnull().any()
#Checking how many columns have null values
df.info()

#Using individual functions to do EDA
#Checking out Statistical data Mean Median Mode correlation
df.mean()
df.median()
df.mode()

#How is the data distributed and detecting Outliers
df.std()
df.max()
df.min()
df.quantile(0.25)*1.5
df.quantile(0.75)*1.5

#How many Outliers in the Total Food ordered column
df.columns
df.dtypes
df.set_index(['Hospital Name'])
df['Total Food ordered'].loc[df['Total Food ordered'] <=238.5].count()
df['Total Food ordered'].loc[df['Total Food ordered'] >=679.5].count()

#Visualizing the dataset
df.boxplot(figsize=(10, 6))
df.plot.box(vert=False)
df.kurtosis()
df.skew()
import scipy.stats as sp
sp.skew(df['Total Food ordered'])

#Visualizing dataset
df.plot()
df.hist(figsize=(10, 6))
df.plot.area()
df.plot.area(stacked=False)


#Now look at correlation and patterns
df.corr()

#Change to dataset columns
df.plot.scatter(x='Total Food Wasted', y='No of Guests with Inpatient',s=df['Total Food Wasted']*2)
df.plot.hexbin(x='Total Food Wasted', y='No of Guests with Inpatient', gridsize=25)

#Change to dataset columns
#Look at crosstabulation to conclude EDA
df.columns
df.dtypes
#Counting the Categorical variables
my_tab = pd.crosstab(index=df["Feedback"], columns="Count")      # Name the count column
my_tab = pd.crosstab(index=df["Type of Hospital"], columns="Count")      # Name the count column

print(my_tab)
my_tab=my_tab.sort_values('Count', ascending=[False])
print(my_tab)
#my_tab.sum()
data_counts = pd.DataFrame(my_tab)
pd.DataFrame(data_counts).transpose().plot(kind='bar', stacked=False)

#Data Preparation Steps
#Step 1 Split data into features and target variable
# Split the data into features and target label
wastage = pd.DataFrame(df['Total Food Wasted'])

dropp=df[['Total Food Wasted','Feedback','Type of Hospital','Total No of beds']]
features= df.drop(dropp, axis=1)
wastage.columns
features.columns

#Step 2 Shuffle & Split Final Dataset
# Import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

# Shuffle and split the data into training and testing subsets
features=shuffle(features,  random_state=0)
wastage=shuffle(wastage,  random_state=0)
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, wastage, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


#Model Building & Evaluation
#Creating the the Model for prediction

#Loading model Libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVC  
 

#Creating Linear Regression object
regr = linear_model.LinearRegression()
linear_svm = LinearSVC().fit(X_train,y_train)

regr.fit(X_train,y_train)
y_pred= regr.predict(X_test)
yy_pred= linear_svm.predict(X_test)
#Printing Codfficients
print('Coefficients: \n',regr.coef_)
print(LinearSVC().fit(X_train,y_train).coef_)
regr.score(X_train,y_train)
#Mean squared error
print("mean squared error:  %.2f" %mean_squared_error(y_test,y_pred))

#Variance score
print("Variance score: %2f"  % r2_score(y_test, y_pred))

#Plot and visualize the Linear Regression plot
plt.plot(X_test, y_pred, linewidth=3)
plt.show()

#Checking graphically the boundaries formed by Linear SVM
line = np.linspace(-15, 15)  
for coef, intercept in zip(linear_svm.coef_, linear_svm.intercept_):  
    plt.plot(line, -(line * coef[0] + intercept) / coef[1])  #HOW DO WE KNOW
plt.ylim(-10, 15)  
plt.xlim(-10, 8)  
plt.show()  

predicted= regr.predict([[820,81,363,35]])
print(predicted)

features.head(2)







