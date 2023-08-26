import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Loading the Titanic dataset
data = pd.read_csv("train.csv")  # Make sure to provide the correct file path
print(data.head())  # Check the first few rows of the dataset

# Preprocessing the data (you might need to handle missing values and encode categorical variables)

# Selecting relevant features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
y = data['Survived']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plotting the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()
