# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Boston Dataset

import pandas as pd

# A data set containing housing values in 506 suburbs of Boston.
#
# - **crim**: per capita crime rate by town.
# - **zn**: proportion of residential land zoned for lots over 25,000 sq.ft.
# - **indus**: proportion of non-retail business acres per town.
# - **chas**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# - **nox**: nitrogen oxides concentration (parts per 10 million).
# - **rm**: average number of rooms per dwelling.
# - **age**: proportion of owner-occupied units built prior to 1940.
# - **dis**: weighted mean of distances to five Boston employment centres.
# - **rad**: index of accessibility to radial highways.
# - **tax**: full-value property-tax rate per 10,000 USD.
# - **ptratio**: pupil-teacher ratio by town.
# - **lstat**: lower status of the population (percent).
# - **medv**: median value of owner-occupied homes in 1000s USD.
#

# This exercise involves the Boston housing data set.

# (a) To begin, load in the Boston data set, which is part of the ISLP
# library.

from ISLP import load_data
Boston = load_data('Boston')
Boston.columns

Boston['chas']=Boston['chas'].astype(bool)

# ### (b) How many rows are in this data set? How many columns? What do the rows and columns represent?

Boston.shape

# - There are 506 rows in the dataset, each one corresponding to on of the suburbs of Boston.
# - There are 13 columns, containing information about different aspects of these suburbs.

# ### (c) Make some pairwise scatterplots of the predictors (columns) in this data set. Describe your findings.

import seaborn as sns

sns.pairplot(data=Boston.iloc[:, : 7])

sns.pairplot(data=Boston[Boston.columns[7:].insert(0, 'crim')])

# ### (d) Are any of the predictors associated with per capita crime rate? If so, explain the relationship.

# nothing very meaningful

# ### (e) Do any of the suburbs of Boston appear to have particularly high crime rates? Tax rates? Pupil-teacher ratios? Comment on the range of each predictor.

for predictor in ['crim', 'tax', 'ptratio']:
    print(f'Range of {predictor}: [{Boston[predictor].min()}, {Boston[predictor].max()}]')

The full property-tax rate and the crim rate seems quite disperse





# ### (f) How many of the suburbs in this data set bound the Charles river?

Boston['chas'].value_counts()

# ### (g) What is the median pupil-teacher ratio among the towns in this data set?

Boston['ptratio'].median()

# ### (h) Which suburb of Boston has lowest median value of owner- occupied homes? What are the values of the other predictors for that suburb, and how do those values compare to the overall ranges for those predictors? Comment on your findings.
#



# ### (i) In this data set, how many of the suburbs average more than seven rooms per dwelling? More than eight rooms per dwelling? Comment on the suburbs that average more than eight rooms per dwelling.




