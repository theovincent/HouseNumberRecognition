"""
This code plots the metric corresponding to the results of the nets on the data sets.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.store_metrics import load_metrics


def show_metrics(root_metric1, axe_legend, root_metric2=""):
    """
    Show the results that has been registered.

    Args:
        root_metric1 (string): the path where the first metric is registered.

        axe_legend (string): the legend of the x_axe.

        root_metric2 (optional)(string): the path where the second metric is registered.
    """
    metric1 = load_metrics(root_metric1)
    second_metric = False
    if root_metric2 != "":
        second_metric = True
        metric2 = load_metrics(root_metric2)

    nb_points = len(metric1)
    epochs = np.arange(1, nb_points + 1, 1, dtype=np.int)

    label1 = root_metric1.parts[-1][: -4]
    plt.plot(epochs, metric1, label=label1)
    if second_metric:
        label2 = root_metric2.parts[-1][: -4]
        plt.plot(epochs, metric2, label=label2)
    plt.legend()
    axes = plt.gca()
    axes.set_xlabel(axe_legend)
    plt.show()


if __name__ == "__main__":
    SAVE_METRICS1 = Path("../../net_data/results/lay3training_accuracy_0.txt")
    SAVE_METRICS2 = Path("../../net_data/results/lay3test_accuracy_0.txt")
    show_metrics(SAVE_METRICS1, "epochs", SAVE_METRICS2)
