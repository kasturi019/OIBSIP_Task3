#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[44]:


import pandas as pd
import numpy as np


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[46]:


df = pd.read_csv(r'C:\Users\Kasturi\Desktop\Oasis\CarPrice.csv')


# In[47]:


df


# In[48]:


df.head()


# In[49]:


df.tail()


# In[50]:


df.info()


# In[51]:


df.describe(include = "all")


# In[52]:


df.isna().sum()


# In[53]:


print(df.fueltype.value_counts())
print(df.aspiration.value_counts())
print(df.doornumber.value_counts())
print(df.carbody.value_counts())
print(df.drivewheel.value_counts())
print(df.enginelocation.value_counts())
print(df.fuelsystem.value_counts())


# In[54]:


plt.subplots(figsize=(40,30))
ax=sns.boxplot(x='CarName', y='price', data= df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()


# In[55]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='fueltype', y='price', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()


# In[56]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='fueltype', y='price', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()


# In[57]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='fueltype', y='price', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()


# In[58]:


plt.subplots(figsize=(15,10))
ax=sns.boxplot(x='stroke', y='price', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()


# In[59]:


plt.subplots(figsize=(15,10))
ax=sns.boxplot(x='drivewheel', y='price', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()


# In[60]:


df = pd.get_dummies(df, columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 
                                   'drivewheel', 'cylindernumber', 'enginelocation', 'enginetype','fuelsystem'])
print(df)


# In[61]:


df


# In[62]:


x = df.drop(['CarName', 'price'], axis = 1)
y = df[['price']]


# In[63]:


x


# In[64]:


y


# In[65]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25, random_state = 50)


# In[66]:


xtrain


# In[67]:


ytrain


# In[68]:


xtest


# In[69]:


ytest


# In[70]:


lin_model = LinearRegression()
lin_model.fit(xtrain, ytrain)


# In[71]:


ypredtest = lin_model.predict(xtest)


# In[72]:


Mean_absolute_error = mean_absolute_error(ytest, ypredtest)
print('Mean_absolute_error:',Mean_absolute_error)


# In[73]:


Mean_squared_error = mean_squared_error(ytest, ypredtest)
print('Mean_squared_error:',Mean_squared_error)


# In[74]:


Mean_squared_error = mean_squared_error(ytest, ypredtest)
print('Mean_squared_error:',Mean_squared_error)


# In[75]:


RSquared = r2_score(ytest, ypredtest)
print('R-Squared:' ,RSquared)


# In[76]:


AdjRsquared = 1-((1-RSquared)*(len(xtest)-1)/(len(xtest)- len(x.columns)-1))
print('AdjRsquared:', AdjRsquared)


# In[77]:


ypredtrain = lin_model.predict(xtrain)


# In[78]:


Mean_absolute_error = mean_absolute_error(ytrain, ypredtrain)
print('Mean_absolute_error:',Mean_absolute_error)


# In[79]:


Mean_squared_error = mean_squared_error(ytrain, ypredtrain)
print('Mean_squared_error:',Mean_squared_error)


# In[80]:


Root_Mean_squared_error = np.sqrt(Mean_squared_error)
print('Root_Mean_squared_error:', Root_Mean_squared_error)


# In[81]:


RSquared = r2_score(ytrain, ypredtrain)
print('R-Squared:' ,RSquared)


# In[82]:


AdjRsquared = 1-((1-RSquared)*(len(xtest)-1)/(len(xtest)- len(x.columns)-1))
print('AdjRsquared:', AdjRsquared)


# In[83]:


x.columns


# In[84]:


ypredtest


# In[85]:


ytest


# In[86]:


ytest - ypredtest


# In[ ]:




