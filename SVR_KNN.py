import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'D:\NIT_DS & AI\8 jan\emp_sal.csv')

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('positive level')
plt.ylabel('salary')
plt.show()


lin_model_pred=lin_reg.predict([[6.5]])
lin_model_pred

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
X_poly=poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(X,y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('polymodel (Polynomial Regression)')
plt.xlabel('positive level')
plt.ylabel('salary')
plt.show()

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


#support vector regresion model
from sklearn.svm import SVR
svr_reg=SVR(kernel='poly',gamma='auto',degree=4)   # Need to check all the parameters and save in excel 
svr_reg.fit(X,y)

svr_reg_pred=svr_reg.predict([[6.5]])
svr_reg_pred

#KNN regressor
from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=2) # Need to check all the parameters and save in excel
knn_reg.fit(X, y)

#Prediction
knn_reg_pred=knn_reg.predict([[6.5]])
knn_reg_pred


#Decision tree we use in classificationd

from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor()
dt_reg.fit(X,y)

#prediction
dt_reg_pred=dt_reg.predict([[6.5]])
dt_reg_pred


#Random Forest

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=27,random_state=0)
rf_reg.fit(X,y)

rf_reg_pred=rf_reg.predict([[6.5]])
rf_reg_pred

#xgboost regressor
import xgboost as xg
xgb_r=xg.XGBRegressor(objective='reg:squarederror',n_estimators=4)
xgb_r.fit(X,y)

xgb_reg_pred=xgb_r.predict([[6.5]])
xgb_reg_pred

















