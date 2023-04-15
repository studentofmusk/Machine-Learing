from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

features = iris.data
lable = iris.target
# print(iris.DESCR)

print(features[0], lable[0])


classifier = KNeighborsClassifier()
classifier.fit(features, lable)

# prediction = classifier.predict([[1, 1, 1, 1]])
# prediction = classifier.predict([[32, 4, 3, 3]])
prediction = classifier.predict([[43, 4, 31, 5]])

print(prediction)





