"""
This code plots the results corresponding to the results of the nets on the data sets.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.store_results import load_results


def show_results(root_results1, axe_legend, root_results2=""):
    """
    Show the results that has been registered.

    Args:
        root_results1 (string): the path where the first results is registered.

        axe_legend (string): the legend of the x_axe.

        root_results2 (optional)(string): the path where the second results is registered.
    """
    results1 = load_results(root_results1)
    second_results = False
    if root_results2 != "":
        second_results = True
        results2 = load_results(root_results2)

    nb_points = len(results1)
    epochs = np.arange(1, nb_points + 1, 1, dtype=np.int)

    label1 = root_results1.parts[-1][: -4]
    plt.plot(epochs, results1, label=label1)
    if second_results:
        label2 = root_results2.parts[-1][: -4]
        plt.plot(epochs, results2, label=label2)
    plt.legend()
    axes = plt.gca()
    axes.set_xlabel(axe_legend)
    plt.show()


if __name__ == "__main__":
    SAVE_RESULTS1 = Path("../../net_data/results/train/lay3training_F_beta_1.txt")
    SAVE_RESULTS2 = Path("../../net_data/results/test/lay3test_F_beta_1.txt")
    show_results(SAVE_RESULTS1, "epochs", SAVE_RESULTS2)
