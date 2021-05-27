import pandas as pd
from sklearn import metrics

data = pd.read_csv("Advertising.csv",index_col=0)

# print(data.head())

# print(data.shape)

# print(data.tail())


# pred = [90,80,10,20]

# error = (10**2,10*2,10**2,20**2)/4

# error = metrics.mean_squared_error(true,pred)]
x_cols = ["TV","Radio"]

X = data[x_cols]

y = data.Sales
# import seaborn as sbn
# import matplotlib.pyplot as plt

# sbn.pairplot(data,x_vars=["TV","Radio","Newspaper"],y_vars=["Sales"],size=7,aspect=0.7)
# plt.show()

# sbn.pairplot(data,x_vars=["TV","Radio","Newspaper"],y_vars=["Sales"],size=7,aspect=0.7,kind="reg")
# plt.show()

#B is beta values
#y = B0 + B1*x1 + B2*x2 +....+Bn*xn
#y is the response value
#B0 is the intercept fit() 
#B1 is the coeffecient of x1 fit()
#Bn is the coeffectient of xn

from sklearn.linear_model import LinearRegression

Lr = LinearRegression()

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y)

Lr.fit(X_train,y_train)

#three methods-
#MAE - Mean Absoulute Error
# true = [100,90 ,20,0]
# pred = [90,80,10,20]

# error = (10,10,10,20)/4

# error = metrics.mean_absolute_error(true,pred)

#MSE - Mean Squared Error
# true = [100,90 ,20,0]
# pred = [90,80,10,20]

# error = (10**2,10*2,10**2,20**2)/4

# error = metrics.mean_squared_error(true,pred)

#RMSE - Root Mean Squared Error

import numpy as np

# true = [100,90 ,20,0]
# pred = [90,80,10,20]

# error = np.sqrt((10**2,10*2,10**2,20**2)/4)

# error = np.sqrt(metrics.mean_squared_error(true,pred))

#Mean Sqared error popular as it 'large' error punish
#root mean sqared error more popular as it is interpretable as 'y' value 

y_pred = Lr.predict(X_test)

error = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

print(error)