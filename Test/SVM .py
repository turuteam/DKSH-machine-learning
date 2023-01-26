


import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd




# Load the CSV file
data = pd.read_csv("Test Code.csv")





data.head(500)





data.info()



X = data[['normal', 'spam']].values
y = data['phishing'].values


# In[33]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[34]:



# Create an SVM model
clf = svm.SVC(kernel='linear')


# In[35]:


# Train the model on the training data
clf.fit(X_train, y_train)


# In[36]:


# Test the model on the test data
y_pred = clf.predict(X_test)


# In[ ]:


# Print the accuracy of the model
acc = np.mean(y_pred == y_test)
print("Accuracy:", acc)

# In this case, the accuracy is 1.0, which means that the model was able to correctly predict the labels for all of the test data. This is considered to be a perfect score, as it means that the model made no errors in its predictions. However, it is important to note that a high accuracy score alone does not necessarily mean that the model is good, as it could be overfitting or other factors may be considered.
# In[ ]:




