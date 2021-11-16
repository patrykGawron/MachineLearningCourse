import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import pyplot as plt

diabetes = pd.read_csv("dataset_37_diabetes.csv")
diabetes['class'].loc[diabetes['class']=="tested_negative"] = 0
diabetes['class'].loc[diabetes['class']=="tested_positive"] = 1
diabetes['class'] = diabetes['class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(['class'], axis=1),
                                                    diabetes['class'], random_state=42,
                                                    stratify=diabetes["class"],
                                                    test_size=0.25)

classifiers = [KNeighborsClassifier(), DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5)]
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

for stat in ["plas","pres","skin","insu","mass","pedi","age"]:
    X_train[stat].loc[X_train[stat]==0] = np.NaN
    print(f"a:{stat}", X_train[stat].isna().sum()/len(X_train) * 100)

for classifier in classifiers:
    classifier.fit(X_train.dropna(), y_train[X_train.notna().all(axis=1)])
    print(classifier.score(X_test.dropna(), y_test[X_test.notna().all(axis=1)]))

imputer_clf = [
    ("KNN", Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', MinMaxScaler()),
        ('clasifier', KNeighborsClassifier())
    ])),
    ("DTC", Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', MinMaxScaler()),
        ('clasifier', DecisionTreeClassifier(max_depth=5))
    ])),
    ("RFC", Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', MinMaxScaler()),
        ('clasifier', RandomForestClassifier(max_depth=5))
    ]))
]
print("===========================================")
for name, classifier in imputer_clf:
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

zscore = abs((diabetes - diabetes.mean())/diabetes.std())
diabetes = diabetes.loc[~(zscore>=3).any(axis=1)]

from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.01)
clf.fit(diabetes[["mass", "plas"]])
print(clf.predict(diabetes[["mass", "plas"]].head()))

# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(np.array(diabetes[["mass", 'plas']]),
#                       np.array(clf.predict(diabetes[["mass", "plas"]])),
#                       clf)
# plt.show()

from sklearn import svm, model_selection
import pandas as pd
import seaborn as sns

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"],
                                                    iris['target'],
                                                    random_state=42,
                                                    stratify=iris['target'],
                                                    test_size=0.25)

parameters = {
    'kernel': ('linear', 'rbf', 'sigmoid'),
    'C':[1, 10, 30]
}
clf = GridSearchCV(svm.SVC(), parameters, cv=10)
clf.fit(X_train, y_train)

pvt = pd.pivot_table(
    pd.DataFrame(clf.cv_results_),
    values='mean_test_score',
    index='param_kernel',
    columns='param_C'
)

# ax = sns.heatmap(pvt)
# plt.show()

import pickle

with open("model.wzium", "wb") as file:
    pickle.dump(clf.best_estimator_, file)

print(clf.best_estimator_.score(X_test, y_test))

with open("model.wzium", "rb") as file:
    clf = pickle.load(file)
print(clf.score(X_test, y_test))
