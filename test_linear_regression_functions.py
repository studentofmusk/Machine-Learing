from sklearn import linear_model
from sklearn.metrics import mean_squared_error 
import numpy as np 
import matplotlib.pyplot as plt


x = np.array([[1], [2], [3]])
x_train = x
x_test = x

y = np.array([3, 2, 4])

y_train =y
y_test = y

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)
# print(y_predicted)

print("Mean square Error:", mean_squared_error(y_test, y_predicted))
print("Weights:", model.coef_)
print("Intecepts:", model.intercept_)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_predicted)
plt.show() 