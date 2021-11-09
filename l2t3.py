from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
rain_to_number = {
    "brak": 0,
    "małe": 1,
    "średnie": 2,
    "duże": 3
}

X_train = [
    [-30, 23, "duże"],
    [20, 15, "średnie"],
    [10, 3, "małe"],
    [15, 8, "brak"],
    [1, 9, "średnie"],
    [23, 3, "brak"],
    [18, 12, "duże"],
    [17, 11, "małe"],
    [19, 19, "małe"],
    [25, 10, "średnie"],
]
y_train = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]

X_test = [
    [-30, 22, "małe"],
    [22, 1, "brak"],
    [23, 3, "średnie"],
    [7, 8, "brak"],
    [1, 9, "duże"],
    [13, 12, "duże"],
    [30, 12, "brak"],
    [26, 9, "małe"],
    [35, 13, "brak"],
    [20, 19, "małe"]
]

y_test = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]

for i in range(len(X_train)):
    X_train[i][2] = rain_to_number[X_train[i][2]]
    X_test[i][2] = rain_to_number[X_test[i][2]]

svc = DecisionTreeClassifier()
svc.fit(X_train, y_train)
scr = svc.score(X_test, y_test)
print(scr)
print(svc.predict([[-30, 23, 3]]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for (temperature, time, rain), output in zip(X_train, y_train):
    if output == 0:
        ax.scatter(temperature, time, rain, c='b')
    else:
        ax.scatter(temperature, time, rain, c='r')

plt.show()

