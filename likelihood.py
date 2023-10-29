#importing required libraries

import numpy as np
import matplotlib.pyplot as plt

#generating random values of x and y
np.random.seed(0)
x = 2 * np.random.rand(100, 1)  # Random feature
y = 3 + 2 * x + np.random.randn(100, 1)  # Linear relationship with noise

#print(x)
#print(y)

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

# Print the maximum likelihood estimate
print("Maximum Likelihood Estimate (MLE):")
print("Intercept:", beta[0][0])
print("Slope:", beta[1][0])


#plotting the values
plt.plot(x, y, 'o')
plt.plot(x_predict, y_predict, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
