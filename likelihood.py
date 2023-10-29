#importing required libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#generating random values of x and y
np.random.seed(0)
x = 2 * np.random.rand(100, 1)  # Random feature
y = 3 + 2 * x + np.random.randn(100, 1)  # Linear relationship with noise

#print(x)
#print(y)

# Calculate mean and standard deviation (MLE estimates)
mean_mle = np.mean(y)
std_dev_mle = np.std(y)

#concatenating the matrix x with 1s ie adding 1s to the matrix
x_b = np.c_[np.ones((100, 1)), x]
#print(x_b)

#finding the normal equation
beta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(beta)

#predicting values for:
x_predict = np.array([[0.0], [2.0]])
x_predict_b = np.c_[np.ones((2, 1)),  x_predict]
y_predict = x_predict_b.dot(beta)

# Generate data points for the normal distribution curve
x_range = np.linspace(min(y), max(y), num=1000)

# Calculate the probability density function (PDF) using MLE parameters
pdf_values = norm.pdf(x_range, mean_mle, std_dev_mle)

# Print the maximum likelihood estimate
print("Maximum Likelihood Estimate (MLE):")
print("Intercept:", beta[0][0])
print("Slope:", beta[1][0])

# Plot the histogram of the data
plt.hist(y, bins=20, density=True, alpha=0.6, color='g', label='Data Histogram')

# Plot the normal distribution curve using MLE parameters
plt.plot(x_range, pdf_values, 'r', label='Normal Distribution Curve (MLE)')

plt.xlabel('y')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Histogram and Normal Distribution Curve of Generated Data')


#plotting the values
plt.plot(x, y, 'o')
plt.plot(x_predict, y_predict, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

