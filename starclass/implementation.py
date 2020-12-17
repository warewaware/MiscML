#!/usr/bin/env python3

"""Classifies stars based on temperature,
luminosity, radius, absolute magnitude, and apparent color
with a simple application of linear discriminant analysis.

We find the model has a mean accuracy of 98.7% with a std
deviation of 1.4 percentage points. We also find that the
presence of the spectral class attribute in the original
dataset lower the accuracy of 1%, likely due to the
dependence on the temperature and luminosity attributes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Function to split data into train and test sets
from sklearn.model_selection import train_test_split
# Tools to analyze model accuracy
from sklearn.metrics import accuracy_score

data = pd.read_csv("~/repos/LDA156FinalProj/source/data/star_data.csv")
print(data)


spectra_map = {'O':5, 'B':4, 'A':3, 'F':2, 'G':1, 'K':0, 'M':-1}
color_map = { 'Red':-5, 'Orange-Red':-4, 'Orange':-3, 'Pale yellow orange':-2, 
             'Yellow-White':-1, 'Whitish':0, 'White':1,
             'Blue-White': 2, 'Blue':2}

data.replace(spectra_map,inplace=True)
data.replace(color_map, inplace=True)

# X, y = data.drop(columns=['Star type']), data['Star type']
X, y = data.drop(columns=['Spectral Class','Star type']), data['Star type']

# Define train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, 
                                                    random_state=0)


# Initialize model
model = LDA(solver='eigen',shrinkage=None)
# Fit training set
model.fit(X_train, y_train)

# Load predicted classes and analyze performance
predicted = model.predict(X_test)

print("Model accuracy:", accuracy_score(y_test, predicted))
def LDA_error(X, y, train_size, random_state):
  # Define train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      train_size=train_size, 
                                                      random_state=random_state)
  model = LDA(solver='eigen',shrinkage=None).fit(X_train, y_train)
  return accuracy_score(y_test, model.predict(X_test))

n_iter = 1000
errors = [LDA_error(X, y, train_size=0.7, random_state=i) 
          for i in range(n_iter)]

print(f"Model accuracy over {n_iter} splits")
print("Mean:   ", np.mean(errors))
print("Std dev:", np.std(errors))

plt.scatter(range(n_iter), errors)
plt.title(f"LDA model accuracy on {n_iter} data splits")
plt.xlabel("Random split")
plt.ylabel("Testing accuracy")
plt.show()
