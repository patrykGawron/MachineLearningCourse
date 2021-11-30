import mlflow.sklearn
import pandas as pd
import random
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


class Lab5:

    def __init__(self):
        self.titanic_dataset = pd.read_csv("../Titanic.csv")
        # self.todo1()
        self.todo2()
        self.todo6()
        self.todo4()
        self.todo7()
        self.todo3()
        self.todo8()
        # self.todo5()
        # self.todo9()
        self.todo()

    def todo1(self):
        self.titanic_dataset.info()
        print(self.titanic_dataset.describe())

    def todo2(self):
        self.titanic_dataset.drop(columns=["boat", "body", "home.dest", "cabin",
                                           "ticket", "name"], axis=1,
                                  inplace=True)
        self.titanic_dataset.replace(to_replace="?", value=np.NaN, inplace=True)

    def todo3(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.titanic_dataset.drop(columns=["survived"], axis=1),
            self.titanic_dataset.loc[:, ["survived"]], random_state=42, test_size=0.1
        )

    def todo4(self):
        correct = 0
        for i, row in self.titanic_dataset.iterrows():
            survived = self.survived()
            correct += survived == row["survived"]
        print(f"Accurac {correct / len(self.titanic_dataset)}")

    def todo5(self):
        msno.bar(self.X_train)
        plt.show()

    def todo6(self):
        self.titanic_dataset["embarked"].iloc[168] = "S"
        self.titanic_dataset["embarked"].iloc[284] = "S"
        self.titanic_dataset.drop(1225, inplace=True)
        self.titanic_dataset["age"] = pd.to_numeric(self.titanic_dataset["age"],
                                                    downcast="float")
        # self.titanic_dataset.loc[self.titanic_dataset["pclass"]==3, 'age'].hist()
        self.ages_titanic = self.titanic_dataset.groupby(
            ["sex", "pclass"])["age"].median().round(1)
        print(self.ages_titanic)

        for row, passenger in self.titanic_dataset.loc[
            np.isnan(self.titanic_dataset["age"])].iterrows():
            self.titanic_dataset["age"].iloc[row] = self.ages_titanic[
                passenger["sex"]][passenger["pclass"]]
            if self.titanic_dataset["age"].iloc[row] == np.NaN:
                print(passenger.sex, passenger.pclass)
                print(self.ages_titanic[passenger.sex][passenger.pclass])
        self.titanic_dataset.dropna(inplace=True)

    def todo7(self):
        self.titanic_dataset["sex"].loc[self.titanic_dataset["sex"] == "female"] = 1
        self.titanic_dataset["sex"].loc[self.titanic_dataset["sex"] == "male"] = 0

        self.titanic_dataset["pclass"].loc[self.titanic_dataset["pclass"] == 3] = 0
        self.titanic_dataset["pclass"].loc[self.titanic_dataset["pclass"] == 2] = 0.5

        self.titanic_dataset["sex"] = self.titanic_dataset["sex"].astype(float)
        self.titanic_dataset["pclass"] = self.titanic_dataset["pclass"].astype(float)
        self.titanic_dataset["fare"] = self.titanic_dataset["fare"].astype(float)

        self.titanic_dataset["family_size"] = self.titanic_dataset["sibsp"] + \
                self.titanic_dataset["parch"]
        self.titanic_dataset["family_size"] = self.titanic_dataset["family_size"]\
            .astype(float)
        self.titanic_dataset.drop(["sibsp", "parch", "embarked"], axis=1, inplace=True)

    def todo8(self):
        self.clf = RandomForestClassifier()
        self.clf.fit(self.X_train, self.y_train.values.ravel())
        self.clf.score(self.X_test, self.y_test)

    def survived(self):
        val = random.randrange(0, 1)
        if val >= 0.5:
            return True
        else:
            return False

    def todo9(self):
        param_grid= {"n_estimators": [1, 10, 100, 1000],
             "max_depth": [None, 1, 10, 100],
             "class_weight": ["balanced", None]}

        self.gs = GridSearchCV(self.clf, param_grid)
        self.gs.fit(self.X_train, self.y_train.values.ravel())
        print(self.gs.best_params_)

    def todo(self):
        import time
        mlflow.sklearn.autolog()
        clfs = [RandomForestClassifier(), SVC(), LinearRegression()]
        for clf in clfs:
            with mlflow.start_run(run_name=type(clf).__name__):
                start_time = time.perf_counter()
                clf.fit(self.X_train, self.y_train)
                duration = time.perf_counter() - start_time
                scr = clf.score(self.X_test, self.y_test)
                mlflow.log_metric("score", scr)
                mlflow.log_metric("duration", duration)


if __name__ == "__main__":
    lab = Lab5()
