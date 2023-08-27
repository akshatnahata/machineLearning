from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample training data
X = [
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1]
]

# Corresponding class labels
y = ['A', 'B', 'A', 'B', 'A']

# Sample testing data
X_test = [
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]

# True class labels for testing data
y_test_true = ['A', 'B']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
