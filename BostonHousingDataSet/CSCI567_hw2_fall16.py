'''
Created on October 2, 2016

@author: arpita
'''
import RidgeRegression
import LinearRegression


LinearRegression.run_LinearRegression()
LinearRegression.run_LR_polynomial_features()

print '\n\n**************************************************************'
RidgeRegression.run_ridge_estimator()
RidgeRegression.ridge_estimator_CV()
