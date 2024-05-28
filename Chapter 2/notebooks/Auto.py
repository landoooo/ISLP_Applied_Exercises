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

# # 2.1 Overview of Statistical Learning - Auto dataset

# 9. This exercise involves the Auto data set studied in the lab. Make sure that the missing values have been removed from the data.

# ### (a) Which of the predictors are quantitative, and which are qualitative?

import pandas as pd

# !pwd

auto = pd.read_csv('data/Auto.csv')

auto

auto.dtypes

auto['horsepower_num'] = auto['horsepower'].apply(pd.to_numeric, errors='coerce')

auto[auto['horsepower_num'].isna()]

auto['cylinders'].value_counts()



# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### (b) What is the range of each quantitative predictor? You can answer this using the min() and max() methods in numpy.
#
# -



# ### (c) What is the mean and standard deviation of each quantitative predictor?
# .min()
# .max()
#



# ### (d) Now remove the 10th through 85th observations. What is the range, mean, and standard deviation of each predictor in the subset of the data that remains?
#



# ### (e) Using the full data set, investigate the predictors graphically, using scatterplots or other tools of your choice. Create some plots highlighting the relationships among the predictors. Comment on your findings.
#



# ### (f) Suppose that we wish to predict gas mileage (mpg) on the basis of the other variables. Do your plots suggest that any of the other variables might be useful in predicting mpg? Justify your answer.

#




