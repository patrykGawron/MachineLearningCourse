from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 1, 1, 1]

clf = DecisionTreeClassifier()
clf.fit(X, y)

print(clf.predict([[1, 1], [1, 0], [0, 1], [0, 0]]))   # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.
plot_tree(clf)
plt.show()