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

# +
import pandas as pd 

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# !pwd
# -

# From https://islp.readthedocs.io/en/latest/datasets/Auto.html 
#
# - **mpg**: miles per gallon
# - **cylinders**: Number of cylinders between 4 and 8
# - **displacement**: Engine displacement (cu. inches)
# - **horsepower**: Engine horsepower
# - **weight**: Vehicle weight (lbs.)
# - **acceleration**: Time to accelerate from 0 to 60 mph (sec.)
# - **year**: Model year (modulo 100)
# - **origin**: Origin of car (1. American, 2. European, 3. Japanese)
# - **name**: Vehicle name

# 9. This exercise involves the Auto data set studied in the lab. Make sure that the missing values have been removed from the data.

# ### (a) Which of the predictors are quantitative, and which are qualitative?

auto = pd.read_csv('data/Auto.csv')

auto

auto.dtypes

auto['horsepower'] = auto['horsepower'].apply(pd.to_numeric, errors='coerce')

auto[auto['horsepower'].isna()]

auto['cylinders'].value_counts()

# ##### Qualitative variables: 
# - cylinders
# - year
# - origin
# - name
# ##### Quantitative variables: 
# - mpg
# - displacement
# - horsepower
# - weight
# - acceleration

# ### (b) What is the range of each quantitative predictor? You can answer this using the min() and max() methods in numpy.
# ``
# .min()
# ``
#
# ``
# .max()
# ``
#

# +
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_predictors = auto.select_dtypes(include=numerics).columns

for predictor in numeric_predictors:
    'Range of {}: [{}, {}]'.format(predictor, auto[predictor].min(), auto[predictor].max())
# -

# ### (c) What is the mean and standard deviation of each quantitative predictor?

for predictor in numeric_predictors:
    print(f'{predictor}: [Mean: {auto[predictor].mean():.2f}, Std: {auto[predictor].std():.2f}]')

# ### (d) Now remove the 10th through 85th observations. What is the range, mean, and standard deviation of each predictor in the subset of the data that remains?
#

# +
auto_d = auto.drop(auto.index[10:85])

for predictor in numeric_predictors:
    print(f'{predictor}: [Mean: {auto_d[predictor].mean():.2f}, Std: {auto_d[predictor].std():.2f}]')
# -

# ### (e) Using the full data set, investigate the predictors graphically, using scatterplots or other tools of your choice. Create some plots highlighting the relationships among the predictors. Comment on your findings.
#

pd.plotting.scatter_matrix(auto[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'year', 'horsepower']], figsize=(12,12));

auto['brand'] = [x.split()[0] for x in auto['name']]

auto[['brand', 'name']]

auto['brand'].value_counts()

# +
brands_dict = {
    'vw': 'volkswagen'
    ,'vokswagen': 'volkswagen'
    ,'chevy': 'chevrolet'
    ,'maxda': 'mazda'
    ,'mercedes': 'mercedes-benz'
    ,'chevroelt': 'chevrolet'
    ,'toyouta': 'toyota'
    ,'vokswagen': 'volkswagen'
    ,'vokswagen': 'volkswagen'
}

origin_dict = {
    1: 'America'
    ,2: 'Europe'
    ,3: 'Japan'
}
# -

auto['brand'] = auto['brand'].replace(brands_dict)
auto['origin'] = auto['origin'].replace(origin_dict)

auto.dtypes

auto['brand'] = auto['brand'].astype("category")
auto['origin'] = auto['origin'].astype("category")

auto

# +
auto_america = auto[auto['origin']=='America']
auto_europe = auto[auto['origin']=='Europe']
auto_japan = auto[auto['origin']=='Japan']

auto_japan
# -

pd.plotting.scatter_matrix(auto_japan[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'year', 'horsepower']], figsize=(12,12));

pd.plotting.scatter_matrix(auto_america[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'year', 'horsepower']], figsize=(12,12));

pd.plotting.scatter_matrix(auto_europe[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'year', 'horsepower']], figsize=(12,12));

# ### (f) Suppose that we wish to predict gas mileage (mpg) on the basis of the other variables. Do your plots suggest that any of the other variables might be useful in predicting mpg? Justify your answer.

# cylinders, displacement, weight,year, origin and horsepower.

auto.boxplot(column=['mpg'], by='brand', figsize=(12,12))  


import seaborn as sns

import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(12,12))
sns.boxplot(data=auto, hue='brand', y='mpg', )


