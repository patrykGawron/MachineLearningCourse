import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

from utils.battery_problem_reader import get_battery_data

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC


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
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)

    plt.scatter(np.array(X)[:, 0],
                np.array(X)[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.show()
    # X_train = X_train[['sepal length (cm)', 'sepal width (cm)']]
    # X_test = X_test[['sepal length (cm)', 'sepal width (cm)']]


    # skaler = StandardScaler()
    # skaler.fit(X_train)
    #
    # print(y_train.value_counts() / len(y_train) * 100)
    # X_train = skaler.transform(X_train)
    plt.scatter(np.array(X_train)[:, 0],
                np.array(X_train)[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.show()

    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC())
    ])

    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
    from sklearn.naive_bayes import GaussianNB
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', GaussianNB())
    ])

    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))


    # Plotting decision regions


    classfiers = ["LogisticRegression", "SVC",
                  "DecisionTreeClassifier", "RandomForestClassifier"]

    for classfier in classfiers:
        train_test_classifier(classfier, X_train, y_train, X_test, y_test)


def train_test_classifier(classfier_name, X_train, y_train, X_test, y_test):
    clf = globals()[classfier_name]()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    # plot_decision_regions(np.array(X_test), np.array(y_test), clf=clf, legend=1)
    #
    # # Adding axes annotations
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel('petal length [cm]')
    # plt.title(f'{classfier_name} on Iris')
    # plt.show()

def main():
    todo_2()


if __name__ == "__main__":
    main()
