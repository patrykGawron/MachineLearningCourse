import pandas as pd
from sklearn.model_selection import train_test_split
import random


class Lab5:

    def __init__(self):
        self.titanic_dataset = pd.read_csv("../Titanic.csv")
        self.todo1()
        self.todo2()
        self.todo3()
        self.todo4()

    def todo1(self):
        self.titanic_dataset.info()
        print(self.titanic_dataset.describe())

    def todo2(self):
        self.titanic_dataset.drop(columns=["boat", "body", "home.dest"])

    def todo3(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.titanic_dataset.drop(columns=["survived"]),
            self.titanic_dataset.loc[:, ["survived"]], random_state=42, test_size=0.1
        )
        print(self.titanic_dataset.drop(columns=["survived"]))

    def todo4(self):
        correct = 0
        for i, row in self.titanic_dataset.iterrows():
            survived = self.survived()
            correct += survived == row["survived"]
        print(f"Accurac {correct / len(self.titanic_dataset)}")

    def survived(self):
        val = random.randrange(0, 1)
        if val >= 0.5:
            return True
        else:
            return False


if __name__ == "__main__":
    lab = Lab5()
