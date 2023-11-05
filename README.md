NAME:GORDON ODHIAMBO

REG NUMBER: I39/2687/2020

GROUP 2 MEMBERS:

1:POLINE NTWIGA: I39/2672/2020

2:JOSHUA MOCHAMA: I39/2688/2020



NORMAL EQUATION METHOD FOR LINEAR REGRESSION
********************************************

This basically is the matrix derivaties in the working out of the least square cost function

Cost function refers to the equation involving the least sum of the square of residuals of the fit line

On normal Equation method,  the equation [((X^T*X)*X^T))*Y] Provides the Y-intercept and The Slope of The Least Square Cost Function

Remember as demonstated by the code below

X^T=The transpose of X

X and Y = x and y values respectively

PROBABILITY INTERPRETATION OF LINEAR REGRESSION
***********************************************

In the context of linear regression, the maximum likelihood estimation (MLE) is a method used to estimate the parameters of the model, In the probabilistic interpretation of linear regression

           N
L(θ;X,y) = ∏  1/(SQRT(2πσ^2)exp(-((Y^i - m*X^i)^2)/(2πσ^2)
          i=1
To find the MLE estimates for the parameters (m), we maximize the likelihood function. 

It is often more convenient to work with the log-likelihood function, which is the natural logarithm of the likelihood function:
​
​In the case of linear regression, this optimization problem can often be solved analytically using matrix calculus, leading to the well-known closed-form solution for the regression coefficients

LOCALLY WEIGHTED REGRESSION
*******************************

Its the fitting of a curve to a dataset unlike the linear regression.

It uses the concept of Weighted Least Squares to fit in the curve


The file are in python format.I used pycharm IDE to run and test the codes.....any IDE can be used and execute to see the results and analysis of the work.
   
This method still can apply normal equation method thus providing the maximum likelihood estimates

Follow the Python codes for detailed comments and code
