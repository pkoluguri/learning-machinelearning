from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

iris = load_iris()

x = iris.data

y = iris.target

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=4)

knn_1 = KNeighborsClassifier(n_neighbors=1)
knn = KNeighborsClassifier(n_neighbors=5)
Lr = LogisticRegression()

knn_1.fit(x_train,y_train)
knn.fit(x_train,y_train)
Lr.fit(x_train,y_train)

y_pred = knn_1.predict(x_test)
ky_pred = knn.predict(x_test)
Lr_pred = Lr.predict(x_test)

print("1 neighbour accuracy score")
print(accuracy_score(y_test,y_pred))

print("5 neighbour accuracy score")
print(accuracy_score(y_test,ky_pred))

print("Logistic regression accuracy score")
print(accuracy_score(y_test,y_pred))