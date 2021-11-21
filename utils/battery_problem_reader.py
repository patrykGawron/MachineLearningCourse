import csv
import os


def get_battery_data():
    xs: list = []
    ys: list = []
    # TODO: Resolve path construction for reliable file loading
    with open("battery_problem_data.csv", "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x, y = row
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def main():
    pass


if __name__ == "__main__":
    main()
