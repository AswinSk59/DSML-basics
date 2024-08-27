#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


pip install pandas


# In[1]:


import numpy as np 
import pandas as pd 
data = pd.Series([10, 20, 30, 40, 50, 60, 70]) 
data 



# In[3]:


data = pd.Series([10, 20, 30, 40, 50, 60, 80],index = ['a','b','c','d', 'e','f','g'], dtype ='int8') 
data 


# In[6]:


data.values 


# In[7]:


array_data = data.values 
print(array_data) 


# In[8]:


data.index 
data_series = { 'Columnl': pd.Series([100, 200, 300, 400, 500, 600, 700], dtype ='int16'), 'Column2': pd.Series([10, 20, 30, 40, 50, 60, 70], dtype = 'int16') 
} 


data_series


# In[9]:


pd.DataFrame(data_series) 


# In[13]:


movies_df= pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/boston_train.csv', sep = ',') 
movies_df 


# In[14]:


movies_df.head()


# In[15]:


movies_df.tail()


# In[20]:


pip install openpyxl


# In[2]:


import pandas as pd
stock_data = pd.read_excel("https://github.com/ammishra08/MachineLearning/raw/master/Datasets/data_akbilgic.xlsx", header=1)
stock_data 


# In[5]:


movies_df= pd.read_csv('https://raw.githubusercontent.com/ammishra08/MachineLearning/master/Datasets/boston_train.csv', sep = ',') 
movies_df 




# In[10]:


movies_df.shape  


# In[9]:


movies_df.columns  


# In[8]:


len(movies_df.columns) 


# In[11]:


print(movies_df.shape[0], movies_df.shape[1]) 


# In[19]:


data_series = { 'Column1': pd.Series([100, 200, 300, 400, 500, 600], index = ['a','b','c','d','e','f'], dtype = 'int16'), 'Column2': pd.Series([10, 20, 30, 40, 50, 70], index = ['a','b','c','d', 'e','g'], dtype = 'int16') } 
df = pd.DataFrame(data_series) 
df


# In[20]:


df.isnull()


# In[21]:


df.isnull().sum()


# In[22]:


df.isna().sum()


# In[23]:


df.notnull() 
df[df['Column1'].isnull() == True] 


# In[24]:


df[df['Column2'].isnull() == True] 


# In[ ]:




