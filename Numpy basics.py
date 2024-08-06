#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


get_ipython().system('pip install numpy')


# In[1]:


import numpy as np
print(np.__version__)


# In[5]:


import numpy as np
x = np.array([1, 2, 3, 4])
print(x)
print(type(x))
print(x.shape)


# In[18]:


y=np.array([[1,2],[3,4]])
print(y)
print(y.shape)
z= np.array([[1+0.j,2+5.j]])
print(z)
print(z.shape)


# In[3]:


a=np.zeros((2,3))
print(a)
print(a.shape)


# In[7]:


b=np.ones((2,3), dtype=int) 
print(b)


# In[15]:


d = np.eye((3),dtype=int)
print(d)


# In[19]:


e= np.arange(10)
print(e)
e=np.arange(12, 21)
print(e)
e= np.arange(5,20,3)
print(e)


# In[23]:


f=np.linspace(1,20,7) 
print(f)



# In[24]:


g= np.random.random((3,4))
print(g)


# In[28]:


h = np.random.random((3,4)) 
print(h.reshape(2,2,3))


# In[36]:


x= np.arange(12)
print(x) 
print(x[4])
print(x[-1])
x.resize(3,4)
print(x)
print(x[-1,-1])
print(x[2][3])


# In[39]:


y= np.arange(1,26)
print(y)
print(y[:3])
print(y[10:])
print(y[10:15])
print(y[-5:])
print(y[3:-3])
print(y[::3])


# In[40]:


print(y.reshape((5,5)))
print(y)
y=y.reshape((5,5))
print(y)
print(y[:3,:3])
print(y[2:-1,1:-1])


# In[43]:


print(y[:,:-1])
print(y[:,-1])
print(y[::,::2])
print(y)


# In[44]:


print(y[1::2,1::2])


# In[47]:


a= np.arange(1,6)
b= np.arange(6,11)
print(a)
print(b)
print(a+b)
print(a-b)
print(b-a)
print(a**2)
print(a>3)


# In[49]:


a=np.arange(0,4).reshape((2,2))
b = np.eye(2)
print(a*b)
print(np.dot(a,b))
x= np.arange(1,10).reshape(3,3)
print(x)
print(x.sum())

print(x.sum(axis=0))


# In[52]:


print(x.sum(axis=1))


# In[57]:


x= np.arange(1,19).reshape(3,3,2)

print(x)
print(x.sum(axis=1))



# In[58]:


x= np.arange(1,10).reshape(3,3)
print(x)
print(x.max())

print(x.max(axis=0))



# In[59]:


print(x.transpose())


# In[ ]:




