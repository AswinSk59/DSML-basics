# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
# Train the model
mlp.fit(X_train, y_train)
# Predict on the test set
y_pred = mlp.predict(X_test)
# Evaluate the model
print(&quot;Accuracy:&quot;, accuracy_score(y_test, y_pred))
print(&quot;\nClassification Report:\n&quot;, classification_report(y_test, y_pred))
print(&quot;\nConfusion Matrix:\n&quot;, confusion_matrix(y_test, y_pred))
