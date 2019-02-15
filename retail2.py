# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:50:08 2018

@author: PUNEETMATHUR Copyright 2018 All rights reserved
DO NOT COPY WTTHOUT PERMISSION
"""

import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
warnings.filterwarnings("ignore", category = UserWarning, module = "cross_validation")

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################
# Importing libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import pandas.plotting

####################################################
####################################################
######### FUNCTIONS FOR DATA VISUALIZATION #########
####################################################
####################################################
 
def pcaOutput(good_data, pca):
	'''
	Visualizing the PCA results and calculating explained variance
	'''

	# I am doing Dimension indexing through pca components
	dims = dims = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# Creating the PCA components pandas dataframe from the dimensions
	comps = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
	comps.index = dims

	# Calculating PCA explained variance for each component
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dims

	# Creating a bar plot visualization for better understanding
	fig, ax = plt.subplots(figsize = (24,12))

	# Plotting the feature weights as a function of the components
	comps.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dims, rotation=0)


	# Displaying the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Returning back a concatenated DataFrame
	return pd.concat([variance_ratios, comps], axis = 1)

def clusterOutput(redData, preds, centers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions
	Adds points for cluster centers for-selected sample data
	'''

	preds = pd.DataFrame(preds, columns = ['Cluster'])
	plot_data = pd.concat([preds, redData], axis = 1)

	# I am Generating the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Creating the Color map to distinguish between clusters
	cmap = cm.get_cmap('gist_rainbow')

	# Coloring the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):   
	    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);

	# Plotting the centers with cross mark indicators
	for i, c in enumerate(centers):
	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
	               alpha = 1, linewidth = 2, marker = 'o', s=200);
	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);

	# Plotting transformed sample points 
	ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
	           s = 150, linewidth = 4, color = 'black', marker = 'x');

	# Plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids with Numerical markingr\nTick marks denote Transformed Sample Data");


def biPlotter(cleanData, redData, pca):
    '''
    Building a biplot for PCA of the reduced data and the projections of the original features.
    Variable cleanData: original data, before transformation.
    Creating a pandas dataframe with valid column names
    redData: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    This function returns: a matplotlib AxesSubplot object (for any additional customization)
    This function is inspired by the script by Teddy Roland on Biplot in Python:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data    
    ax.scatter(x=redData.loc[:, 'Dimension 1'], y=redData.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, cleanData.columns[i], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("Principal Component plane with original feature projections.", fontsize=16);
    return ax
    

def channelOutput(redData, outliers, pca_samples):
	'''
	Here we are Visualizing the PCA-reduced cluster data in two dimensions using the full dataset
	Data is labeled by "Channel" points added for selected sample data
	'''

	# Check that the dataset is loadable
	try:
	    full_data = pd.read_csv("E:/retail2.csv")
	except:
	    print "Dataset could not be loaded. Is the file missing?"
	    return False

	# Create the Channel DataFrame
	chl = pd.DataFrame(full_data['Channel'], columns = ['Channel'])
	chl = chl.drop(chl.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([redData, chl], axis = 1)
	
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned Channel
	labels = ['Segment 1/Segment 2/Segment3', 'Retail Customer']
	grouped = labeled.groupby('Channel')
	for i, chl in grouped:   
	    chl.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i-1)*1.0/2), label = labels[i-1], s=30);
	    
	# Plot transformed sample points   
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled");

###########################################################
###########################################################
######### END OF FUNCTIONS FOR DATA VISUALIZATION #########
###########################################################
###########################################################



##########################################################################
##########################################################################
#######################  IMPLEMENTATION CODE #############################
##########################################################################
##########################################################################

###########################################
# I am Suppressing matplotlib and other module deprecation user warnings
# This is Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
warnings.filterwarnings("ignore", category = UserWarning, module = "cross_validationb")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np


# Load the retail customers dataset
try:
    data = pd.read_csv("E:\\retail2.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Retail customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
# Display a description of the dataset
display(data.describe())


#   Selecting three indices of my choice from the dataset
indices = [225,182,400]

# Creating a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of Retail customers dataset:"
display(samples)

#lOOKING AT THE PLOTS TO CHECK THE TREND
pd.DataFrame(samples).transpose().plot(kind='bar', stacked=False, figsize=(8,8), title='Retail customers dataset')

# Calculating deviations from Mean and Median
#to check how much the 3 samples deviate from the Center.
delta_mean = samples - data.mean().round()
delta_median = samples - data.median().round()

#display(delta_mean, delta_median)
delta_mean.plot.bar(figsize=(30,10), title='Compared to MEAN', grid=True)
delta_median.plot.bar(figsize=(30,10), title='Compared to MEDIAN', grid=True);

#Importing Libraries
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Broke down the logic into a scoring function and a For loop
def scorer(feature):
    #   Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    relevant_data = data.drop([feature], axis=1)

    #   Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(relevant_data, data[feature], test_size=0.25, random_state=45)

    #   Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=45).fit(X_train, y_train)

    #   Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)

    print("The R2 score for feature {:16} is {:+.5f}".format(feature, score))

for feature in data.columns.values:
    scorer(feature)

# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (24,12), diagonal = 'kde');
#checking Correlation table
data.corr()

#   Scale the data using the natural logarithm
log_data = np.log(data)
#   Scale the sample data using the natural logarithm
log_samples = np.log(samples)
# Display the log-transformed sample data
display(log_samples)
log_samples.corr()

#Identifying Outliers
#Using Counters
from collections import Counter
outliers_counter = Counter()

# For each feature finding out the data points with extreme high or low values - Outliers
outliers_scores = None 

for feature in log_data.keys():
    
    #   Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    #   Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    #   Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    empty = np.zeros(len(log_data[feature]))
    aboveQ3 = log_data[feature].values - Q3 - step
    belowQ3 = log_data[feature].values - Q1 + step
    current_outliers_scores = np.array(np.maximum(empty, aboveQ3) - np.minimum(empty, belowQ3)).reshape([-1,1])
    outliers_scores = current_outliers_scores if outliers_scores is None else np.hstack([outliers_scores, current_outliers_scores])
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    current_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(current_outliers)
    outliers_counter.update(current_outliers.index.values)
    
# Select the indices for data points you wish to remove
min_outliers_count = 2
outliers = [x[0] for x in outliers_counter.items() if x[1] >= min_outliers_count]
print("Data points considered outlier for more than 1 feature: {}".format(outliers))

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

  
#   Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=len(good_data.columns)).fit(good_data)

#   Transform log_samples using the PCA fit above
pca_samples =pca.transform(log_samples)

# Generate PCA results plot
pcaOutput = pcaOutput(good_data, pca)
# show results
display(pcaOutput)
# Cumulative explained variance should add to 1
display(pcaOutput['Explained Variance'].cumsum())

from sklearn.decomposition import PCA
#   Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

#   Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

#   Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Create a biplot
biPlotter(good_data, reduced_data, pca)

#Import GMM and silhouette score library
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score

#Divided the logic into Clustering and Scoring functions
#Then Iterated through the clusters in the identified range

#This function creates Cluster using GMM
def clusterCreator(data, n_clusters):
    clusterer = GMM(n_components=n_clusters, covariance_type='full', random_state=45).fit(data)
    preds = clusterer.predict(data)
    centers = clusterer.means_
    sample_preds = clusterer.predict(pca_samples)
    return preds, centers, sample_preds

#Scoring after creating Clusters
def scorer(data, n_clusters):
    preds, _, _  = clusterCreator(data, n_clusters)
    score = silhouette_score(data, preds)
    return score

#Iterate in the clusters and Print silhouette score
for n_clusters in range(2,10):
    score = scorer(reduced_data, n_clusters)
    print "For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score)

n_clusters=0

# Resetting as due to loop run before we need to reset values again with 2 cluster components
clusterer = GMM(n_components=2).fit(reduced_data)
predictions = clusterer.predict(reduced_data)
centers = clusterer.means_
sample_preds = clusterer.predict(pca_samples)
# Display the results of the clustering from implementation
clusterOutput(reduced_data, predictions, centers, pca_samples)

log_centers = pca.inverse_transform(centers)

#   Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
#Compare the True Centers from the Central values Mean and Median
#display(true_centers - data.mean().round())
delta_mean=true_centers - data.mean().round()
#display(true_centers - data.median().round())
delta_median=true_centers - data.median().round()
delta_mean.plot.bar(figsize=(10,4), title='Compared to MEAN', grid=True)
delta_median.plot.bar(figsize=(10,4), title='Compared to MEDIAN', grid=True);

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred
samples

# Display the clustering results based on 'Channel' data
redData=reduced_data
channelOutput(redData, outliers, pca_samples)

##########################################################################
##########################################################################
#######################  END OF CODE #####################################
##########################################################################
##########################################################################

