import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import datasets

from utils.battery_problem_reader import get_battery_data


def todo_1():
    x, y = get_battery_data()
    x = np.array(x).reshape((-1, 1))

    reg = LinearRegression()
    reg.fit(x, y)
    pred = reg.predict(x)

    model = Pipeline([('poly', PolynomialFeatures(degree=10)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x, y)
    poly_pred = model.predict(x)

    plt.plot(x, y, 'ro')
    plt.plot(x, pred, 'bo')
    plt.plot(x, poly_pred, 'go')
    plt.show()


def todo_2():
    iris = datasets.load_iris(as_frame=True)


def main():
    todo_1()

if __name__ == "__main__":
    main()
