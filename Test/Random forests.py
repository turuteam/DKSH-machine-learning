#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Load the CSV file
data = pd.read_csv("Test Code.csv")


# In[3]:


data.head(1000)


# In[4]:



# Select the 3 features to use in the model
X = data[['normal', 'spam']]

# Select the target column
y = data['phishing']


# In[5]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[6]:



# Initialize the random forest classifier
clf = RandomForestClassifier()


# In[7]:


# Train the model on the training data
clf.fit(X_train, y_train)



# In[8]:


# Make predictions on the test data
y_pred = clf.predict(X_test)


# In[9]:


# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)


# In[10]:


print("Accuracy: ", accuracy)


# In[ ]:




