# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:46:06 2018

@author: PUNEETMATHUR
"""
#Importing python libraries
import pandas as pd
from io import StringIO
import requests
import os
os.getcwd()

#Reading dataset from flat file
fname="Diabetes_Dataset.csv"
patients= pd.read_csv(fname, low_memory=False, index_col=False)
df= pd.DataFrame(patients)

#Look at the first record
print(df.head(1))
#Check the shape size and columns in the dataset
print(df.size)
print(df.shape)
print(df.columns)
df.dtypes
#Check if there are any columns with empty/null dataset
df.isnull().any()
#Checking how many columns have null values
df.info()

dfworking=df.drop('Patient ID',axis=1)
#You can use Describe method to see however since our columns are more 
#We will use individual functions to do EDA
print(dfworking.describe)

#Using individual functions to do EDA
#Checking out Statistical data Mean Median Mode correlation
dfworking.mean()
dfworking.median()
dfworking.mode()


#How is the data distributed and detecting Outliers
dfworking.std()
dfworking.max()
dfworking.min()
dfworking.quantile(0.25)*1.5
dfworking.quantile(0.75)*1.5

#How many Outliers in the BPSystolic column
df.columns
df.dtypes
dfworking.set_index(['BPSystolic'])
dfworking['BPSystolic'].loc[df['BPSystolic'] >=183.562500].count()
dfworking.set_index(['Patient ID'])


dfworking.boxplot(figsize=(10, 6))
dfworking.plot.box(vert=False)
dfworking.boxplot(column=['A1cTEST','BPSystolic','BPDiastolic','HeartRateBPM','BMI','Age'],figsize=(10, 6))
dfworking.kurtosis()
dfworking.skew()
import scipy.stats as sp
sp.skew(dfworking.A1cTEST)

#Visualizing dataset
dfworking.plot()
dfworking.hist(column=['A1cTEST','BPSystolic','BPDiastolic','HeartRateBPM','BMI','Age'],figsize=(10, 6))
dfworking.plot.area()
dfworking.plot.area(stacked=False)


#Now look at correlation and patterns
dfworking.corr()
dfworking.plot.scatter(x='A1cTEST', y='BPSystolic',s=dfworking['A1cTEST']*2)
dfworking.plot.scatter(x='A1cTEST', y='BPSystolic',s=dfworking['BPSystolic']*0.13)
dfworking.plot.hexbin(x='A1cTEST', y='BPSystolic', gridsize=25)

#Look at crosstabulation to conclude EDA
df.columns
#Counting the Categorical variables
my_tab = pd.crosstab(index=df["Gender"], columns="Count")      # Name the count column
my_tab = pd.crosstab(index=df["Type of diabetes"], columns="Count")      # Name the count column
my_tab = pd.crosstab(index=df["Diabetes status"], columns="Count")      # Name the count column
my_tab = pd.crosstab(index=df["FrozenShoulder"], columns="Count")      # Name the count column
my_tab = pd.crosstab(index=df["CarpalTunnelSynd"], columns="Count")      # Name the count column
my_tab = pd.crosstab(index=df["DuputrensCont"], columns="Count")      # Name the count column
print(my_tab)
my_tab=my_tab.sort_values('Count', ascending=[False])
print(my_tab)
my_tab.sum()
data_counts = pd.DataFrame(my_tab)
pd.DataFrame(data_counts).transpose().plot(kind='bar', stacked=False)

#Data Preparation Steps
#Step 1 Split data into features and target variable
# Split the data into features and target label
diabetics = pd.DataFrame(dfworking['Diabetes status'])
features = pd.DataFrame(dfworking.drop('Diabetes status', axis = 1))
diabetics.columns
features.columns

#Step 2 Standardize dataset
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
dfworking.dtypes
numerical = ['Age','A1cTEST','BPSystolic','BPDiastolic','HeartRateBPM','BMI']
features_raw[numerical] = scaler.fit_transform(dfworking[numerical])

# Show an example of a record with scaling applied
display(features_raw[numerical].head(n = 1))

# Step 3 One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)
features.columns

#Checking output
display(features.head(1),diabetics.head(1))

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# see the encoded feature names
print(encoded)

#Step 4 Shuffle & Split Final Dataset
# Import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

# Shuffle and split the data into training and testing subsets
features=shuffle(features,  random_state=0)
diabetics=shuffle(diabetics,  random_state=0)
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, diabetics, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

#Model Building & Evaluation
#Creating the the Model for prediction

#Loading model Libraries
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# prepare models
seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

import warnings
warnings.filterwarnings("ignore")

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


import matplotlib.pyplot as plt
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

