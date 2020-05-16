#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('hiring.csv')
df


# In[3]:


import math
b=df['test_score(out of 10)'].median()


# In[4]:


df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(b)


# In[5]:


df


# In[6]:


df.experience=df.experience.map({'two':2,'three':3,'five':5,'seven':7,'ten':10,'eleven':11})
#or
#df.experience = d.experience.apply(w2n.word_to_num)
df


# In[7]:


df.experience=df.experience.fillna(0)
df


# In[8]:


model=linear_model.LinearRegression()
model.fit(df.drop('salary($)',axis='columns'),df['salary($)'])


# In[9]:


model.predict([[2.0,9.0,6]])


# In[ ]:





# In[ ]:




