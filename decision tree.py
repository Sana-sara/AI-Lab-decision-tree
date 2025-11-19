#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load libraries
from sklearn.tree import DecisionTreeClassifier     # import Decision Tree Classifier
from sklearn.model_selection import train_test_split # import train_test_split
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv('Student-Pass-Fail-Data.csv')

# Prepare the dataset
X = np.array(df.drop('Pass_Or_Fail', axis=1))
y = np.array(df['Pass_Or_Fail'])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Building decision tree model
clf = DecisionTreeClassifier()

# Train decision tree classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
predictions = clf.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])

# Model accuracy
acc = clf.score(X_test, y_test)
print(acc)

# Create decision tree classifier object (fixed capitalization and spelling)
clf = DecisionTreeClassifier(criterion="entropy")

# Train decision tree classifier (corrected variable names)
clf = clf.fit(X_train, y_train)

# Print accuracy again (optional)
print(acc)

# Plot tree
plot_tree(clf)


# In[ ]:




