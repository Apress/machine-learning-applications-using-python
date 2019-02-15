# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 13:30:46 2018

@author: PUNEETMATHUR
"""
#Loading libraries
import pandas as pd

#Loading the retail dataset
survey=pd.read_csv("C:/Puneet/All_DOCS/Prediction_Files/DataScience-Python3_UDEMY_COURSE/DataScience-Python3/ml-100k/aystsaga_002.csv")

#Looking at the dataset one row
print(survey.head(1))
#Shape of the dataset Rows and Columns
print(survey.shape)

#Looking at the columns of the dataset
print(survey.columns)
#What are the datatypes of each column do we need to convert any of them
print(survey.dtypes)
#Are there any Null values in the dataset
print(survey.isnull().any())

#Creating a pivot table for product ratings
productRatings = survey.pivot_table(index=['user_id'],columns=['product_name'],values='rating')
#Peeking into the pivot table
productRatings.head(1)

##===========================Customer chooses a product======================================================
#Sample Product rating view assume here that a customer selected the product 'Dalmere Chocolate cup'
#What recommendation are you going to give based on this selection
#Users who bought 'Dalmere Chocolate cup' also bought:
dalmereRatings = productRatings['Dalmere Chocolate cup']
dalmereRatings.head()

#Now finding products with similar ratings
similarProducts = productRatings.corrwith(dalmereRatings)
#Dropping the NA values to get more meaningful data
similarProducts = similarProducts.dropna()
df = pd.DataFrame(similarProducts)
df.head(10)

similarProducts.sort_values(ascending=False)

#Now let us get rid of spurious results in the similarities
import numpy as np
productStats = survey.groupby('product_name').agg({'rating': [np.size, np.mean]})
productStats.head()

productStats.sort_values([('rating', 'mean')], ascending=False)[:15]

#I am now getting rid of ratings size greater than 50 to create meaning results which matter to the business
popularProducts = productStats['rating']['size'] >= 50
productStats[popularProducts].sort_values([('rating', 'mean')], ascending=False)[:15]
df = productStats[popularProducts].join(pd.DataFrame(similarProducts, columns=['similarity']))
df.head(15)
df.sort_values(['similarity'], ascending=False)[:15]
#Visualizing the Bar plot for similarities
df['similarity'].plot(kind='bar')
ax = df['similarity'].plot(kind='bar') 
x_offset = -0.03
y_offset = 0.02
for p in ax.patches:
    b = p.get_bbox()
    val = "{:+.2f}".format(b.y1 + b.y0)        
    ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

#===========================Customer chooses another product======================================================
#2nd Sample Product rating view
aesoRatings = productRatings['Aeso napkins 10 PACK']
aesoRatings.head()

#Now finding products with similar ratings
similarProducts = productRatings.corrwith(aesoRatings)
similarProducts = similarProducts.dropna()
df = pd.DataFrame(similarProducts)
df.head(10)

similarProducts.sort_values(ascending=False)

#Now let us get rid of spurious results in the similarities
import numpy as np
productStats = survey.groupby('product_name').agg({'rating': [np.size, np.mean]})
productStats.head()

productStats.sort_values([('rating', 'mean')], ascending=False)[:15]

#I am now getting rid of ratings size greater than 20 to create meaning results which matter to the business
popularProducts = productStats['rating']['size'] >= 50
productStats[popularProducts].sort_values([('rating', 'mean')], ascending=False)[:15]
df = productStats[popularProducts].join(pd.DataFrame(similarProducts, columns=['similarity']))
df.head(15)
df.sort_values(['similarity'], ascending=False)[:15]

#Visualization of similarities
df['similarity'].plot(kind='bar')
ax = df['similarity'].plot(kind='bar') 
x_offset = -0.03
y_offset = 0.02
for p in ax.patches:
    b = p.get_bbox()
    val = "{:+.2f}".format(b.y1 + b.y0)        
    ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

