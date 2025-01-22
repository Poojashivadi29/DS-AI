import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\NIT_DS & AI\27 dec\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state=0)

x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test, color="red")
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs Exp(test set)')
plt.xlabel('Years of exp')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope=regressor.coef_
print(m_slope)

c_inter=regressor.intercept_
print(c_inter)

y_15 = m_slope*15 + c_inter
y_20 = m_slope*20 + c_inter
print(y_15)
print(y_20)

print(f"Intercept:{regressor.intercept_}")
print(f"Coefficient:{regressor.coef_}")

comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)
#Statistics for Machine Learning

#introduction for Scipy sklearn

dataset.mean()
dataset.median()
dataset['Salary'].mode()
dataset['Salary'].mean()
dataset.var()
dataset.std()

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])
dataset.skew()
dataset['Salary'].skew()
dataset.sem()



# To caluclate Z-Score we import a Scipy library first

import scipy.stats as stats
dataset.apply(stats.zscore)

#Degree of Freedom
a=dataset.shape[0]
b=dataset.shape[1]
degree_of_freedom=a-b
print(degree_of_freedom)


y_mean=np.mean(y)
ssr=np.sum((y_pred-y_mean)**2)
print(ssr)



y=y[0:6]
sse=np.sum((y-y_pred)**2)
print(sse)


mean_total=np.mean(dataset.values)
sst=np.sum((dataset.values-mean_total)**2)
print(sst)


r_square=1-ssr/sst
print(r_square)



from sklearn.metrics import mean_squared_error


bias =regressor.score(x_train,y_train)
print(bias)

variance =regressor.score(x_test,y_test)
print(variance)







import pickle
filename="regressor.pkl"
with open(filename,'wb') as file:
    pickle.dump(regressor, file)
    print("Model has be pickled and saved as regressor.pkl")



import os
print(os.getcwd())