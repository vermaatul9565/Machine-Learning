#!/usr/bin/env python
# coding: utf-8

# Using $Linear-Regression$ algorithm predict the capita per income of year $2020$.

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns


# In[23]:


df=pd.read_csv('canada_per_capita_income.csv')
df.shape


# In[30]:


df


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("year")
plt.ylabel("per capita income (US$)")
plt.scatter(df.year,df["per capita income (US$)"])
sns.regplot(x='year',y='per capita income (US$)',data=df)


# In[25]:


model=linear_model.LinearRegression()
train_x=df.drop("per capita income (US$)",axis='columns')
train_x.shape


# In[33]:


train_y=df.drop("year",axis='columns')  # 2D
# or 
# train_y=df["per capita income (US$)"] # 1D
train_y.shape


# In[29]:


model.fit(train_x,train_y)


# In[31]:


model.predict([[2020]])


# In[ ]:




