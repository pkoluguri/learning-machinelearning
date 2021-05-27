# predict a continuous response

import pandas as pd
import sklearn
import numpy as np

#reading the data 
data = pd.read_csv("Advertising.csv",index_col=0)

# print(data.head())

# print(data.tail())

# print(data.shape)

#plotting data for better view
# import seaborn as sb
# import matplotlib.pyplot as plt

# sb.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size=7,aspect=0.7)
# plt.show()

# input()

# sb.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size=7,aspect=0.7,kind="reg")
# plt.show()

#B is beta values
#Linear Regression - it tries to find a line that best fits the data after the line is learned than it can be used to make predicitions
#Form -

#y = B0+B1x1+B2+X2+...+BnXn
# - y is the response  
# -B0 is the intercept 
# -B1 is the coeffecient of X1
# -Bn is the coeffecient of Xn 

#in this data
#Form is -
#y = B0 + B1 x TV + B2 x Radio  B3 x Newspaper
#the B values are the model coeffecients. these are learned during the model fitting step
#then the model can be used to make predictions

feature_cols = ['TV','Radio','Newspaper']

X = data[feature_cols]

# print(X.head())
# print(type(X))
# print(X.shape)

#response value
y = data['Sales']

#or 

y = data.Sales 

# print(y.head())
# print(type(y))
# print(y.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=1)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(x_train,y_train)

#B0
# print(linreg.intercept_)
#B1,B2,B3
# print(linreg.coef_)

# for X,c in zip(feature_cols,linreg.coef_):print(X,c)

y_pred = linreg.predict(x_test)

#different way to see accuracy-
#1.MAE - Mean Absoulute Error 
#caluculating the difference between the predicition and the original values
#then dividing them by their length

true = [100,50,30,20]
pred = [90,50,50,30]

result = [10,0,20,10]/4

#or

from sklearn import metrics

mae = metrics.mean_absolute_error(true,pred)


#2.MSE - Mean Squared Error
true = [100,50,30,20]
pred = [90,50,50,30]

result = [10**2,0**2,20**2,10**2]/4

#or 

mse = metrics.mean_squared_error(true,pred)

#3. RMSE - Root Mean Squared Error 

rue = [100,50,30,20]
pred = [90,50,50,30]

result = np.sqrt((10**2,0**2,20**2,10**2)/4)

#or 

mse = np.sqrt(metrics.mean_squared_error(true,pred))

#Mean squared error is popular as it 'punishes' larger errors
#Root Mean Squared Error is more popular because it can be interpretable in 'y' units

accuracy_score = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

print(accuracy_score)