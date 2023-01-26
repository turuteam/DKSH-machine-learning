#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[12]:


data = pd.read_csv('Test Code.csv')


# In[13]:


data.head(1000)


# In[16]:


# Split the data into training and testing sets
X = data[["normal", "spam"]]
y = data["phishing"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[17]:


# Create an instance of the Naive Bayes Classifier
classifier = GaussianNB()


# In[18]:


# Fit the classifier to the training data
classifier.fit(X_train, y_train)


# In[19]:


# Predict the class labels for the test data
y_pred = classifier.predict(X_test)


# In[20]:


# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# he accuracy of the Naive Bayes Classifier will be a value between 0 and 1, where 1 represents perfect accuracy. The accuracy will depend on the quality of the data, the number of features, and the size of the dataset.
