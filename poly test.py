import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\NIT_DS & AI\8 jan\emp_sal.csv")

X = dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)

plt.scatter(X,y,color='red')
plt.plot(X,linear_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_model_pred=linear_reg.predict([[6.5]])
lin_model_pred


# polynomial model (default degree=2) 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y) 
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X,y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('polymodel (polynomial Reg)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

ploy_model_reg=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
ploy_model_reg



























