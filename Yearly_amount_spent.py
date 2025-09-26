
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import statsmodels.api as sm
import pylab 
import scipy.stats as stats


customers = pd.read_csv('Ecommerce Customers')
#print(customers.head())

x = customers[['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']
#print(x.head())
#print(y.head())

xtrain,xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(xtrain, ytrain)
#print(lr.coef_)
#print(lr.score(x,y))

cdf = pd.DataFrame(lr.coef_,x.columns,columns=['Coef'])
#print(cdf)

predictions = lr.predict(xtest)


print('Mean Absolute Error:',mean_absolute_error(ytest, predictions))
print('Mean Squared Error:',mean_squared_error(ytest,predictions))
print('Root Mean Squared Error:',math.sqrt(mean_squared_error(ytest,predictions)))

residuals = ytest - predictions
stats.probplot(residuals, dist="norm", plot = pylab)
#pylab.show()
'''
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()
'''
print("Train R²:", lr.score(xtrain, ytrain))
print("Test R²:", lr.score(xtest, ytest))
