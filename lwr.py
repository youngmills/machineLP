import numpy as np
import matplotlib.pyplot as plt

def  locally_weighted_regression(X, y, query_point, tau=0.1):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    query_point = np.hstack((1, query_point))
    weights = np.exp(-np.sum((X - query_point) ** 2, axis=1) / (2 * tau ** 2))
    theta = np.linalg.inv(X.T.dot(np.diag(weights)).dot(X)).dot(X.T).dot(np.diag(weights)).dot(y)
    predicted_value = query_point.dot(theta)
    return predicted_value

# Generate random sample data
np.random.seed(0)
X = 10 * np.random.rand(100, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Query points for prediction
query_points = np.linspace(0, 10, 100)

# Predict using Locally Weighted Regression for each query point
predicted_values = [locally_weighted_regression(X, y, query_point, tau=0.3) for query_point in query_points]

# Plot the original data and the predictions
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(query_points, predicted_values, color='red', label='Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Locally Weighted Regression')
plt.show()
