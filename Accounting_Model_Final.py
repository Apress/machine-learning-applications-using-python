# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:03:51 2018

@author: PUNEETMATHUR
"""

#Importing python libraries
import pandas as pd
from io import StringIO
import requests
import os
import numpy as np
os.getcwd()

#Creating Visualization Functions which will be used throughout this model
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['Amount','Month', 'Fiscal Year']):
        ax = fig.add_subplot(1, 3, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()
#End of Distribution Visualization function


# Plotting Feature Importances through this function
def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(4), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(4) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()      
   
#End of Feature Importances function

#Loading the Dataset
fname="C:/DATASETS/data.ct.gov/PaymentsDataset.csv"
openledger= pd.read_csv(fname, low_memory=False, index_col=False)

#Verify the data Loaded into memory
print(openledger.head(1))

#Loding into Dataframe for easier computation
data= pd.DataFrame(openledger)

#Look at the first record
print(data.head(1))
#Check the shape of columns in the dataset
print(data.shape)
print(data.columns)
data.dtypes

#Data Cleanup
#Check if there are any columns with empty/null dataset
data.isnull().any()


#No Data Cleanup needed so skipping decision to drop the na value rows since they are very less
#data=data.dropna()
data.isnull().sum()

#Total number of records
n_records = len(data.index)

#Number of records where payments are below 1.5 times of Lower Quantile- Lower Outlier Limit

l=data[data['RedFlag'] == 2].index
n_greater_quantile = len(l)

#Number of records where payments are above 1.5 times of upper Quantile- Upper Outlier Limit
l=data[data['RedFlag'] == 1].index
n_lower_quantile = len(l)

#Percentage of Payments above Upper Outlier limit
p=float(n_greater_quantile)/n_records*100.0
greater_percent =p

#Percentage of Payments above Lower Outlier limit
p=float(n_lower_quantile)/n_records*100.0
lower_percent =p

# Print the results
print "Total number of records: {}".format(n_records)
print "High value Payments above 1.5 times of 75th Percentile: {}".format(n_greater_quantile)
print "Low value Payments below 1.5 times of 25th Percentile: {}".format(n_lower_quantile)
print "Percentage of high value Payments: {:.2f}%".format(greater_percent)
print "Percentage of low value Payments: {:.2f}%".format(lower_percent)

# PREPARING DATA
# Split the data into features and target label
payment_raw = pd.DataFrame(data['RedFlag'])
type(payment_raw)
features_raw = data.drop('RedFlag', axis = 1)
#Removing redundant columns from features_raw dataset
features_raw.dtypes
features_raw=features_raw.drop('TransactionNo', axis=1)
features_raw=features_raw.drop('Department', axis=1)
features_raw=features_raw.drop('Account', axis=1)
features_raw=features_raw.drop('Expense Category', axis=1)
features_raw=features_raw.drop('Vendor ID', axis=1)
features_raw=features_raw.drop('Payment Method', axis=1)
features_raw=features_raw.drop('Payment Date', axis=1)
features_raw=features_raw.drop('Invoice ID', axis=1)
features_raw=features_raw.drop('Invoice Date', axis=1)
features_raw=features_raw.drop('Unnamed: 0', axis=1)
features_raw.dtypes
type(features_raw)

# Visualize skewed continuous features of original data
distribution(data)

# Log-transform the skewed features
#Replacing Null values with zero due to software data entry problem
#Known issue in software user screen takes null values there is no check.
import warnings
warnings.filterwarnings("ignore")
features_raw.isnull().sum()

skewed = ['Amount','Month', 'Fiscal Year']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
features_raw.isnull().sum()
features_raw.fillna(0, inplace=True)
features_raw.dtypes

# Visualize the new log distributions
distribution(features_raw, transformed = True)

#Normalizing Numerical Features
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = [ 'Amount','Month', 'Fiscal Year']
#features_raw[numerical] = scaler.fit_transform(data[numerical])
features_raw[numerical] = scaler.fit_transform(features_raw[numerical])
distribution(features_raw)

# Look at record with scaling applied to see if everything is good
display(features_raw.head(n = 1))
features_raw.columns
#Implementation: Data Preprocessing
# One-Hot Encoding 
data.columns
data.dtypes
data['Payment Status']
#data['Expense Category'] #drop from features_raw
#data['Payment Method'] #drop from features_raw

# Encoding the 'Non-Numeric' data to numerical values
#Payment Status column
d={"Paid-Reconciled": 0, "Paid-Unreconciled": 1}
features_raw['Payment Status'] = features_raw['Payment Status'].map(d)

# Printing the number of features after one-hot encoding
encoded = list(features_raw.columns)
print "{} total features after one-hot encoding.".format(len(encoded))


# Importing train_test_split
from sklearn.cross_validation import train_test_split
payment_raw.columns
# Splitting the 'features' and 'RedFlags' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw, payment_raw, test_size = 0.2, random_state = 0)
X_train.columns
X_test.columns
y_train.columns
y_test.columns
# Showing the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#Checking to see if there are any empty rows
X_train.isnull().any()
y_train.isnull().any()
X_train.isnull().sum()
#Evaluating Model Performance
#Establishing Benchmark performance indicator Naive Bayes
#Naive Predictor Performace
from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
NB = GaussianNB()
NB.fit(X_train,y_train)
pred = NB.predict(X_test)

#Calculating Accuracy Score
#Calculating Beta Score
accuracy_score(y_test,pred)
print(f1_score(y_test,pred, average="macro"))
print(precision_score(y_test, pred, average="macro"))
print(recall_score(y_test, pred, average="macro"))  

# I AM GOING TO DO INITIAL MODEL EVALUATION NOW
# Importing the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(max_features=0.2, max_depth=2, min_samples_split=2,random_state=0)
clf_C = LogisticRegression(random_state=0)
clf_D = SGDClassifier(loss="hinge", penalty="l2")
clf_E = ExtraTreesClassifier(n_estimators=2, max_depth=2,min_samples_split=2, random_state=0)
clf_F = RandomForestClassifier(max_depth=2)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
#Defining function since percent is required 3 times


# Collect results on the learners
learners=["Naive Bayes","Decision Tree","Logistic Regression","SGD Classifier","ExtaTrees Classifier","RandomFores Classifier"]
cnt=0
columns=['learner','train_time','pred_time','acc_train','acc_test','f1_score']
learningresults= pd.DataFrame(columns=columns)
results = {}

for learner in [clf_A, clf_B, clf_C,clf_D,clf_E,clf_F]:
    #print(learners[cnt])
    results['learner']=learners[cnt]
    #Fitting the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train, y_train)
    #Calculating the total prediction time
    end = time() # Get end time
    results['train_time'] =  end - start
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(y_train, predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    beta=0.5
    results['f1_score'] = f1_score(y_test,pred, average="macro")
    print(results)
    learningresults.loc[cnt]=results
    cnt=cnt+1

#Looking at the plots to determine the best Classifier for our Dataset
print(learningresults)
learningresults.columns
learningresults.plot(kind='bar', x='learner', legend='reverse',  title='Classifier Algorithms Compared- Accounts Payment Dataset',figsize=(10,10), fontsize=20)
#learningresults.plot(kind='bar', x='learner').savefig('E:/BUSINESS/APRESS/ApplicationsOfMachineLearning/Chapter15/learner_performance.png')

#--------------------------MODEL TUNING ------------------------
#Now I will implement Model tuning
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from IPython.display import display
import pickle, os.path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

def getscore(y_true, y_predict):
    return fbeta_score(y_true, y_predict, beta)

best_clf = None
beta=0.5

#Initialize the classifier

clf_C = LogisticRegression(random_state=0)

# Create the parameters list you wish to tune
#parameters = {'n_estimators':range(10,20),'criterion':['gini','entropy'],'max_depth':range(1,5)}
parameters = {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'C':range(1,10),'max_iter':range(50,100)}

# Make an fbeta_score scoring object
scorer = make_scorer(getscore)

# Perform grid search on the classifier using 'scorer' as the scoring method
#grid_obj = GridSearchCV(clf_C, parameters, scoring=scorer)


#do something
grid_obj = GridSearchCV(clf_C, parameters)

#grid_obj = GridSearchCV(clf_C, parameters)

# Fit the grid search object to the training data and find the optimal parameters   
from datetime import datetime
startTime = datetime.now()

grid_fit = grid_obj.fit(X_train, y_train)
CV_lr = GridSearchCV(estimator=clf_C, param_grid=parameters, cv= 5)
CV_lr.fit(X_train, y_train)
print(datetime.now() - startTime)

    # Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf_C.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5,average='micro'))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5,average='micro'))

# Print the final parameters
df = pd.DataFrame(grid_fit.grid_scores_).sort_values('mean_validation_score').tail()
display(df)
print "Parameters for the optimal model: {}".format(clf.get_params())

# Now Extracting Feature Importances
# mporting a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import fbeta_score

# Training the supervised model on the training set 
model = ExtraTreesClassifier()
model.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
feature_plot(importances, X_train, y_train)

# Feature Selection
# Import functionality for cloning a model
from sklearn.base import clone
best_clf= clf_F
# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)
best_predictions = best_clf.predict(X_test)
# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, average="macro", beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5, average='macro'))

# Print the final parameters

print "Parameters for the optimal model: {}".format(clf.get_params())