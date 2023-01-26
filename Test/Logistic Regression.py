

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Load the CSV file
data = pd.read_csv("Test Code.csv")


# In[3]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[["normal", "spam"]], data["phishing"], test_size=0.2)


# In[4]:


# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[5]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[6]:


# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)





# In[ ]:




