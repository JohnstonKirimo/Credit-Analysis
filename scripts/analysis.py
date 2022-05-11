#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


merchant_category = pd.read_csv('./data/merchant_category_mapping.csv')
merchant_name = pd.read_csv('./data/merchant_name_mapping.csv')
perf_data = pd.read_csv('./data/performance_dataset.csv')


# #### Taking a look at the data

# ##### Overview of the Merchant Category

# In[3]:


merchant_category.head()


# ##### summary info about the merchant_category dataset

# In[4]:


#dimensions
merchant_category.shape


# In[5]:


#get more info on dataset
merchant_category.info()


# ##### summary info about the merchant_name dataset

# In[6]:


#dimensions
merchant_name.shape


# In[7]:


merchant_name.head()


# ##### summary info about the performance dataset

# In[8]:


perf_data.shape


# In[9]:


perf_data.head()


# ##### Note: Data Cleaning is skipped as there are very few missing records in the category and subcategory columns and the rest of the dataset is quite clean

# #### Merging the datasets

# In[10]:


merchant_category_name = merchant_category.merge(merchant_name, how='left', on='merchant_id')
merchant_category_perf = merchant_category_name.merge(perf_data, how='left', on='merchant_id')
merchant_category_perf.head()


# #### Calculating difference in actual vs. predicted repayment

# In[11]:


#get difference between actual and predicted repayment
merchant_category_perf['repayment_diff'] = merchant_category_perf['actual_repayment_pct'] - merchant_category_perf['predicted_repayment_pct']
merchant_category_perf.sample(5)


# In[12]:


avg_repayment_diff = merchant_category_perf['repayment_diff'].agg(['mean'])
avg_repayment_diff


# #### Checking for correlations

# In[13]:


#check correlation between average term and predicted repayment
merchant_category_perf['avg_term'].corr(merchant_category_perf['predicted_repayment_pct'])


# In[14]:


#correlation between term and actual repayment
merchant_category_perf['avg_term'].corr(merchant_category_perf['actual_repayment_pct'])


# In[15]:


#check correlation between all the variables

import seaborn as sns
corr = merchant_category_perf.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.1f', linewidths=.5, ax=ax)


# #### Calculating difference in amounts of loan applied vs approved

# In[16]:


merchant_category_perf['loan_diff'] = merchant_category_perf['avg_auth_amt'] - merchant_category_perf['avg_loan_amt']
merchant_category_perf.sample(5)


# In[17]:


#Get avg loan difference
merchant_category_perf.loan_diff.mean()


# #### Correlation between loan_diff and average fico score

# In[18]:


score_and_loan = merchant_category_perf.plot(x='avg_fico', y='loan_diff', style='rx')


# In[19]:


#some summary statistics of some key variables
merchant_category_perf[['actual_repayment_pct','predicted_repayment_pct','num_trxn','avg_auth_amt','avg_loan_amt','avg_fico','avg_term','avg_apr']].agg(['min','max','mean','median','std'])


# #### Investigating where actual repayment exceeds predicted

# In[20]:


#where does actual repayment exceeds predicted?
pay_diff = merchant_category_perf[merchant_category_perf.actual_repayment_pct >= merchant_category_perf.predicted_repayment_pct]
pay_diff.head(3)


# In[21]:


pay_diff.category.value_counts(normalize=True)


# #### Investigating categories where actual repayment is less than predicted

# In[22]:


#where does actual repayment exceeds predicted?
merchant_category_perf[merchant_category_perf.actual_repayment_pct < merchant_category_perf.predicted_repayment_pct].category.value_counts(normalize=True)


# #### Note:
# - No noticeable difference in distribution of categories where actual repayment exceeds predicted repayment vs where predicted exceeds actual 
# - Therefore, let's look at distribution of overall volume by category
# - Categories with higher volume could imply higher demand and for those with low volume, low demand
# - The business could increase loan incentives for categories or subcategories with higher loan demand as well as categories struggling with low demand

# In[23]:


#distribution of all categories by volume
merchant_category_perf.category.value_counts(normalize = True)


# In[24]:


merchant_category_perf.category.value_counts(normalize = True).plot(kind='bar')
plt.show()


# In[25]:


top_sub_cat = merchant_category_perf.loc[merchant_category_perf.category.isin(['OTHER','JEWELERY','HOME_FURNISHINGS'])]
top_sub_cat.subcategory.value_counts(normalize=True)


# In[26]:


top_sub_cat.subcategory.value_counts(normalize=True).plot(kind='bar')
plt.show()


# #### Note:
# - Of the top three categories, only 4 subcategories make up 60% of the entire volume

# #### Investigating categories with lowest volume

# In[27]:


low_sub_cat = merchant_category_perf.loc[merchant_category_perf.category.isin(['BEAUTY','MENS_FASHION'])]
low_sub_cat.subcategory.value_counts(normalize=True)


# #### Note: 
# - In the categories with the lowest volume, there are only two subcategories, with equal distribution
# - The business could drop these two categories
# - Alternatively, the business could try and boost volume in these struggling categories, as there are only two of them

# #### Investigating term of loan

# In[28]:


merchant_category_perf.avg_term.agg(['min','mean','median','max','std'])


# In[29]:


#Investigating categories where average term is higher than mean

long_loans = merchant_category_perf[merchant_category_perf.avg_term >9.24]
long_loans.category.value_counts(normalize=True)


# #### Do merchants with longer term loans have a higher repayment than those with short term loans?

# In[30]:


round(long_loans.actual_repayment_pct.mean(),2)


# In[31]:


short_loans = merchant_category_perf[merchant_category_perf.avg_term <9.24]
round(short_loans.actual_repayment_pct.mean(),2)


# #### Note: 
# - Merchants who take long term loans have a 3% higher average repayment rate than those with short term loans
