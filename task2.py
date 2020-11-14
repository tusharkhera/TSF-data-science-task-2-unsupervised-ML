#!/usr/bin/env python
# coding: utf-8

# # TUSHAR KHERA
# TSF DATA SCIENCE AND BUSINESS ANALYTICS TASK-2
# (Prediction using unsupervised ML)

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


# loading dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


# In[3]:


#exploring dataset
df.info()


# In[4]:


df.describe()


# In[5]:


#finding k for kMeans implementation
x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
y = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    y.append(kmeans.inertia_)
    
# Plotting the results 
plt.plot(range(1, 11), y)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('y') # Within cluster sum of squares
plt.show()


# In[6]:


# from result , using k=3
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[7]:


#plotting targets
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'pink', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'purple', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'blue', label = 'Iris-virginica')

# Plotting centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], marker='*',
            s = 100, c = 'red', label = 'Centroids')

plt.legend()


# # Result
# optimum number of clusters = 3

# In[ ]:




