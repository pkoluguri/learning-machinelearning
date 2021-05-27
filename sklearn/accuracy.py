from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

iris = load_iris()

x = iris.data

y = iris.target

knn_1 = KNeighborsClassifier(n_neighbors=1)
knn = KNeighborsClassifier(n_neighbors=5)
Lr = LogisticRegression()

#training the data
knn_1.fit(x,y)
knn.fit(x,y)
Lr.fit(x,y)

from sklearn.metrics import accuracy_score

y_pred = knn_1.predict(x)
ky_pred = knn.predict(x)
Lr_pred = Lr.predict(x)

print("1 neighbour accuracy score")
print(accuracy_score(y,y_pred))

print("5 neighbour accuracy score")
print(accuracy_score(y,ky_pred))

print("Logistic regression accuracy score")
print(accuracy_score(y,y_pred))