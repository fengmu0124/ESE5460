import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
ds = fetch_california_housing()

# Data matrix X and target vector Y
X = ds.data
Y = ds.target

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Calculate w* using the formula (X^T * X)^(-1) * X^T * Y
w_star = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train

# Calculate b* using the formula b = (1^T * (Y - X * w)) / n
b_star = (Y_train - X_train @ w_star).mean()

# Predict the target values for training and testing sets
Y_train_pred = X_train @ w_star + b_star
Y_test_pred = X_test @ w_star + b_star

# Calculate training error and testing (validation) error
train_error = np.mean((Y_train - Y_train_pred) ** 2)
test_error = np.mean((Y_test - Y_test_pred) ** 2)

# Print the results
print("w*:", w_star)
print("b*:", b_star)
print("Training error:", train_error)
print("Validation error:", test_error)
