#testing accuracy can vary a lot
from scipy.sparse.construct import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data

y = iris.target

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=4)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

# print(accuracy_score(y_test,y_pred))

#the solution is to split the data bunch of times and average the score
#this is called KFold cross validation
#basically split test training k no of time (ex:k = 5 or k = 3)
#recommended k is 10 (it has been tested)
#diagram
#https://www.google.com/imgres?imgurl=http%3A%2F%2Fethen8181.github.io%2Fmachine-learning%2Fmodel_selection%2Fimg%2Fkfolds.png&imgrefurl=http%3A%2F%2Fethen8181.github.io%2Fmachine-learning%2Fmodel_selection%2Fmodel_selection.html&tbnid=_VTBRT7rqL-rRM&vet=12ahUKEwjt3_aRkufwAhVuKLcAHWuWDEAQMygAegUIARDLAQ..i&docid=52U3U5MFdpM0oM&w=1146&h=689&q=k%20fold%20cross%20validation&ved=2ahUKEwjt3_aRkufwAhVuKLcAHWuWDEAQMygAegUIARDLAQ


#instead of splitting rows it splits columns
#ex-
# row   Training data       Testing data
#  1    4,5,6,7,8,9          1,2,3
#  2    10,11,12,13,17,18    14,15,17
#  3    18,19,23,24,25,26    20,21,22 

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy',)
# print(scores)

#averaging the scores
# print(scores.mean())

k_range = range(1,31)
k_scores = []
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy',)
  k_scores.append(scores.mean())
#print(scores)

# plt.plot(k_range,k_scores)
# plt.xlabel("Value of K")
# plt.ylabel("Cross-Validated accuraccy")
# plt.show()

#automating the above process
from sklearn.model_selection import GridSearchCV


k_range = (1,31)

param_grid =  dict(n_neighbors=k_range)

grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')

grid.fit(X,y)

print(grid.cv_results_)