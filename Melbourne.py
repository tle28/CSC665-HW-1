
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[20]:


csv_df = pd.read_csv("Melbourne_housing_FULL.csv")


# In[21]:


from sklearn.ensemble import RandomForestRegressor


# In[22]:


rf = RandomForestRegressor()


# In[23]:


csv_df.head().T


# In[24]:


feat_df = csv_df.drop('Price', axis=1)


# In[25]:


csv_df.shape


# In[26]:


feat_df.shape


# In[27]:


csv_df['Price']


# In[28]:


y = csv_df['Price'].values


# In[29]:


y


# In[30]:


y.shape


# In[32]:


csv_df.isnull()


# In[33]:


csv_df.isnull().shape


# In[34]:


csv_df.isnull().any(axis=1)


# In[35]:


csv_df.isnull().any(axis=0)


# In[37]:


rows_labeled_na = csv_df.isnull().any(axis=1)


# In[39]:


rows_with_na = csv_df[rows_labeled_na]


# In[42]:


rows_with_data = csv_df[~rows_labeled_na]


# In[43]:


csv_df.shape, rows_with_na.shape, rows_with_data.shape


# In[44]:


feat_df = rows_with_data.drop('Price', axis=1)


# In[46]:


feat_df.shape


# In[47]:


y = rows_with_data['Price'].values


# In[49]:


y.shape


# In[58]:


suburbs={}
feat_df['Suburb']
feat_df['Suburb'].values
for s in feat_df['Suburb'].values:
    if s not in suburbs:
        suburbs[s] = len(suburbs)


# In[59]:


suburbs


# In[95]:


len(suburbs)
feat_df['Suburb'] = feat_df['Suburb'].replace(suburbs)


# In[65]:


feat_df['Suburb']


# In[71]:



feat_df['Address'].astype('category')
feat_df['Address'].astype('category').cat.categories


# In[72]:


feat_df['Type'].astype('category').cat.categories


# In[74]:


feat_df['Type'].astype('category').cat.codes


# In[75]:


feat_df['Type'] = feat_df['Type'].astype('category').cat.codes


# In[79]:


feat_df.head()


# In[77]:


feat_df['Address'] = feat_df['Address'].astype('category').cat.codes
feat_df['Method'] = feat_df['Method'].astype('category').cat.codes
feat_df['SellerG'] = feat_df['SellerG'].astype('category').cat.codes
feat_df['CouncilArea'] = feat_df['CouncilArea'].astype('category').cat.codes
feat_df['Regionname'] = feat_df['Regionname'].astype('category').cat.codes


# In[78]:


feat_df.head()


# In[89]:


feat_df['Date'] = pd.to_datetime(feat_df['Date'], infer_datetime_format=True)
feat_df['Date'] = feat_df['Date'].astype(np.int64)
feat_df.head()


# In[96]:


rf.fit(feat_df, y)

