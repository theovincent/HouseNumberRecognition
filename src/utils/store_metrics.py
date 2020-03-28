"""
This code can be used to store or load the results corresponding to the result of the nets.
"""
from pathlib import Path
import numpy.random as rd
import numpy as np


def store_metrics(table, name_path):
    """
    Store the metric in a .txt file.

    Args:
        table (array, one dimension): the array to be stored.

        name_path (string): the complete path where the table will be stored.
    """
    with open(name_path, 'w') as file:
        for item in table:
            string_row = "{}".format(item)
            string_row += "\n"
            file.write(string_row)


def load_metrics(name_path):
    """
    Read the metric at the end of the path.

    Args:
        name_path (string): the complete path where the table is stored.

    Returns:
        table (array, one dimension): the array to stored at the end of the path.
    """
    table = []
    with open(name_path, 'r') as file:
        for line in file.readlines():
            line = line.split(" ")
            table.append(np.float(line[0]))
    return np.array(table)


if __name__ == "__main__":
    ACCURACY = rd.randint(1, 4, 9)
    SAVE_ROOT = Path("../../net_data/results/test.txt")
    store_metrics(ACCURACY, SAVE_ROOT)
    LOSS = load_metrics(SAVE_ROOT)
    print(LOSS)
