#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


# In[3]:



# Load the CSV file
data = pd.read_csv("Test Code.csv")

# Split the data into features and labels
X = data[['normal', 'spam']]
y = data['phishing']


# In[4]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Define the parameters to be searched
param_grid = {'max_depth': [1, 2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3, 4, 5]}


# In[6]:


# Create the GridSearchCV object
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)


# In[7]:



# Fit the classifier to the data
clf.fit(X_train, y_train)


# In[8]:


# Make predictions on the test data
y_pred = clf.predict(X_test)


# In[9]:


# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)


# In[10]:


print("Accuracy:", accuracy)


# In[11]:


# Print the best parameters
print("Best Parameters:", clf.best_params_)


# In[ ]:




