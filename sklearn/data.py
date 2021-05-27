from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data

y = iris.target

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x,y)

# print(knn.predict([[3.,4.,5.,7.]]))

knn_1 = KNeighborsClassifier(n_neighbors=1)

knn_1.fit(x,y)

# print(knn.predict([[3.,4.,5.,6.]]))

Lr = LogisticRegression()

Lr.fit(x,y)

# print(Lr.predict([[2.,3.,4.,5.]]))

#train and test data on the same dataset method
from sklearn.metrics import accuracy_score

# print("Knn 5 neighbours score:")
# print(accuracy_score(y,knn.predict(x)))

# print("Knn 1 neighbour score:")
# print(accuracy_score(y,knn_1.predict(x)))

# print("Logistic Regression score:")
# print(accuracy_score(y,Lr.predict(x)))

#spit train test method - more accurate than first method
from sklearn.model_selection import train_test_split

x_train , x_test , y_train ,y_test = train_test_split(x,y)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
knn_1.fit(x_train,y_train)

print("Knn 1 neighbour score:")
print( accuracy_score( y_test , knn_1.predict(x_test) ) )

knn.fit(x_train,y_train)

print("Knn 5 neighbours score:")
print( accuracy_score( y_test , knn.predict(x_test) ) )

Lr.fit(x_train,y_train)

print("Logistic Regression score:")
print( accuracy_score( y_test , Lr.predict(x_test) ) )
