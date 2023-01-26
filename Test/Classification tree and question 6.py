#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[10]:


# Load the CSV file
data = pd.read_csv("Test Code.csv")


# In[11]:


# Split the data into features and labels
X = data[['normal', 'spam']]
y = data['phishing']


# In[12]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Train the decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[14]:


# Make predictions on the test data
y_pred = clf.predict(X_test)


# In[15]:



# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[16]:


# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# We first import the necessary libraries (pandas, DecisionTreeClassifier, train_test_split, and accuracy_score) and load the CSV file into a dataframe. We then split the data into features (X) and labels (y) and use the train_test_split function to divide the data into training and testing sets. We then train the decision tree using the training data and make predictions on the test data. Finally, we use the accuracy_score function to calculate the accuracy of the model.

# In[ ]:




