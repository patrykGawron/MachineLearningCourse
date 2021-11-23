import pandas as pd
import random
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

        self.titanic_dataset["famili_size"] = self.titanic_dataset["sibsp"] + \
                self.titanic_dataset["parch"]
        self.titanic_dataset["famili_size"] = self.titanic_dataset["famili_size"]\
            .astype(float)
        self.titanic_dataset.drop(["sibsp", "parch", "embarked"], axis=1, inplace=True)

    def todo8(self):
        self.clf = RandomForestClassifier()
        self.clf.fit(self.X_train, self.y_train)
        self.clf.score(self.X_test, self.y_test)

    def survived(self):
        val = random.randrange(0, 1)
        if val >= 0.5:
            return True
        else:
            return False


if __name__ == "__main__":
    lab = Lab5()
