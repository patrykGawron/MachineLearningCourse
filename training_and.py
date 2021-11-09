from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split

# X = [[0, 0],
#      [0, 1],
#      [1, 0],
#      [1, 1]]
# y = [0, 1, 1, 1]
#
# clf = DecisionTreeClassifier()
# clf.fit(X, y)
#
# print(clf.predict([[1, 1], [1, 0], [0, 1], [0, 0]]))   # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.
# plot_tree(clf)
# plt.show()

data = []
with open("training_data.txt", 'r') as file:
     reader = csv.reader(file, delimiter=',')
     for row in reader:
          data.append(row)
data = np.array(data)
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, random_state=42)
clf = Pipeline([
     ('poly', PolynomialFeatures(degree=2)),
     ('line', LinearRegression())
])

clf.fit(X_train, y_train)
scr = clf.score(X_test, y_test)
print(scr)
# plt.scatter(X_train, clf.predict(X_train))
# plt.show()