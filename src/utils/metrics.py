"""
This module computes several metrics during the training.

Metrics:
average accuracy (average of each class accuracy)
F beta score (with beta = 0.5)
accuracy
loss

classes:
    Metrics
"""
import numpy as np


class Metrics:
    """
    This class computes different metrics with the results.
    """
    def __init__(self, nb_epochs):
        """
        Construct the different table to register the results.

        Args:
            nb_epochs (int): the number of epochs during the training.
        """
        self.nb_epochs = nb_epochs
        self.nb_elements = 0

        # Current metrics
        self.accuracy = np.zeros(10)
        # For each class we register the numerator and the denominator
        self.precision = np.zeros((10, 2))
        self.recall = np.zeros((10, 2))
        self.general_accuracy = 0
        self.loss = 0

        # Saved metrics
        self.saved_average_accuracy = np.zeros(nb_epochs)
        self.saved_fbeta_score = np.zeros(nb_epochs)
        self.saved_accuracy = np.zeros(nb_epochs)
        self.saved_loss = np.zeros(nb_epochs)

    def update(self, estimated_label, true_label, loss):
        """
        Update the table which are registering the results.

        Args:
            estimated_label (int): the label that is guessed by the net.

            true_label (int): the true label of the image.

            loss (int): the loss between the estimated label and the true label.
        """
        # If the net is right
        if estimated_label == true_label:
            # In Every class is correct for the accuracy
            self.accuracy += 1
            self.precision[estimated_label, 0] += 1
            self.recall[true_label, 0] += 1
            self.general_accuracy += 1
        # It the net is wrong
        else:
            # Every class is correct except the class label and true_label
            self.accuracy += 1
            self.accuracy[estimated_label] -= 1
            self.accuracy[true_label] -= 1

        self.precision[estimated_label, 1] += 1
        self.recall[true_label, 1] += 1
        self.nb_elements += 1
        self.loss += loss

    def save(self, epoch):
        """
        Save the metrics at the end of each epoch.

        Args:
            epoch (int): the index of the epoch.
        """
        # Save elements
        self.saved_average_accuracy[epoch] = sum(self.accuracy) / (10 * self.nb_elements)

        avg_precision = compute_average(self.precision)
        avg_recall = compute_average(self.recall)
        if (0.5 ** 2 * avg_precision + avg_recall) == 0:
            self.saved_fbeta_score[epoch] = 0
        else:
            fbeta_score = (1 + 0.5 ** 2) * avg_precision * avg_recall / (0.5 ** 2 * avg_precision + avg_recall)
            self.saved_fbeta_score[epoch] = fbeta_score

        self.saved_accuracy[epoch] = self.general_accuracy / self.nb_elements

        self.saved_loss[epoch] = self.loss / self.nb_elements

        # Clear counters
        self.nb_elements = 0
        self.accuracy = np.zeros(10)
        self.precision = np.zeros((10, 2))
        self.recall = np.zeros((10, 2))
        self.general_accuracy = 0
        self.loss = 0

    def get(self):
        """
        Give the metrics registered during the training.

        Returns:
            saved_average_accuracy (array, shape=(nb_epochs)): the average of each class accuracy
                for each epoch.

            saved_fbeta_score (array, shape=(nb_epochs)): the f_beta score for each epoch.

            saved_accuracy (array, shape=(nb_epochs)): the accuracy for each epoch.

            saved_loss (array, shape=(nb_epochs)): the loss for each epoch.
        """
        return self.saved_average_accuracy, self.saved_fbeta_score, self.saved_accuracy, self.saved_loss


def compute_average(matrix):
    """
    Compute the average of the matrix that has a particular shape:
    the first colon is the numerator, the second one is the denominator,
    each line represent one element.
    We want to compute the average of the elements.

    Ars:
        matrix (array, shape=(any, 2)): the matrix that we want to compute the average.

    Return:
        (float): the average of the matrix.

    >>> mat = np.array([[1, 0], [1, 1]])
    >>> compute_average(mat)
    0.5
    """
    nb_line = matrix.shape[0]
    average = 0

    for index_element in range(nb_line):
        if matrix[index_element, 1] != 0:
            average += matrix[index_element, 0] / matrix[index_element, 1]

    return average / nb_line


if __name__ == "__main__":
    # -- Doc tests -- #
    import doctest
    doctest.testmod()
