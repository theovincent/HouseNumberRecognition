"""
This file executes unittests.
I do not pretend that it tests all our code.
"""
import unittest
import numpy.random as rd
import numpy as np
from src.utils.metrics import Metrics


class MetricsTest(unittest.TestCase):
    """
    Unittest tests for the class Metrics.
    """
    def setUp(self):
        """
        Set random values to the number of epochs and to two different labels.
        """
        # Set a random number of epochs
        self.nb_epochs = rd.randint(1, 20)

        # We take two different labels
        self.label1 = rd.randint(0, 10)
        self.label2 = rd.randint(0, 10)
        self.loss = rd.random()

        while self.label1 == self.label2:
            self.label2 = rd.randint(0, 10)

    def test_update(self):
        """
        Test the method update on two situations:
            - the net is right
            - the net is wrong
        """
        # Construct a metrics
        metrics_rigth = Metrics(self.nb_epochs)
        metrics_wrong = Metrics(self.nb_epochs)

        # ---- If the net is right ---- #
        metrics_rigth.update(self.label1, self.label1, self.loss)

        self.assertEqual(metrics_rigth.nb_elements, 1)

        # Every label is right
        self.assertListEqual(list(metrics_rigth.accuracy), [1] * 10)

        # Only the estimated label has to change
        self.assertTrue(np.array_equal(metrics_rigth.precision[self.label1], np.ones(2)))
        self.assertTrue(np.array_equal(metrics_rigth.precision[: self.label1], np.zeros((self.label1, 2))))
        self.assertTrue(np.array_equal(metrics_rigth.precision[self.label1 + 1:], np.zeros((10 - self.label1 - 1, 2))))

        # Only the true label has to change
        self.assertTrue(np.array_equal(metrics_rigth.recall[self.label1], np.ones(2)))
        self.assertTrue(np.array_equal(metrics_rigth.recall[: self.label1], np.zeros((self.label1, 2))))
        self.assertTrue(np.array_equal(metrics_rigth.recall[self.label1 + 1:], np.zeros((10 - self.label1 - 1, 2))))

        self.assertEqual(metrics_rigth.general_accuracy, 1)
        self.assertEqual(metrics_rigth.loss, self.loss)

        # ---- If the net is wrong ---- #
        metrics_wrong.update(self.label1, self.label2, self.loss)

        self.assertEqual(metrics_wrong.nb_elements, 1)

        # Every label is right except the estimated label and the true label
        expected_accuracy = np.ones(10)
        expected_accuracy[self.label1] = 0
        expected_accuracy[self.label2] = 0
        self.assertTrue(np.array_equal(metrics_wrong.accuracy, expected_accuracy))

        # Only the estimated label has to change
        self.assertTrue(np.array_equal(metrics_wrong.precision[self.label1], np.array([0, 1])))
        self.assertTrue(np.array_equal(metrics_wrong.precision[: self.label1], np.zeros((self.label1, 2))))
        self.assertTrue(np.array_equal(metrics_wrong.precision[self.label1 + 1:], np.zeros((10 - self.label1 - 1, 2))))

        # Only the true labal has the change
        self.assertTrue(np.array_equal(metrics_wrong.recall[self.label2], np.array([0, 1])))
        self.assertTrue(np.array_equal(metrics_wrong.recall[: self.label2], np.zeros((self.label2, 2))))
        self.assertTrue(np.array_equal(metrics_wrong.recall[self.label2 + 1:], np.zeros((10 - self.label2 - 1, 2))))

        self.assertEqual(metrics_wrong.general_accuracy, 0)
        self.assertEqual(metrics_wrong.loss, self.loss)

    def test_save(self):
        """
        Test the method save on two situations:
            - the net is right
            - the net is wrong
        """
        # Construct a metrics
        metrics = Metrics(self.nb_epochs)

        # ---- If the net is right ---- #
        metrics.update(self.label1, self.label1, self.loss)
        metrics.save(0)

        # Verify the saved results
        self.assertEqual(metrics.saved_average_accuracy[0], 1)
        self.assertEqual(metrics.saved_fbeta_score[0], 0.1)
        self.assertEqual(metrics.saved_accuracy[0], 1)
        self.assertGreater(metrics.saved_loss[0], 0.1)

        # Verify that the attributes are cleaned when save is called
        self.assertTrue(np.array_equal(metrics.accuracy, np.zeros(10)))
        self.assertTrue(np.array_equal(metrics.precision, np.zeros((10, 2))))
        self.assertTrue(np.array_equal(metrics.recall, np.zeros((10, 2))))
        self.assertEqual(metrics.general_accuracy, 0)
        self.assertEqual(metrics.loss, 0)

        # ---- If the net is wrong ---- #
        metrics.update(self.label1, self.label2, self.loss)
        metrics.save(0)

        # Verify the saved results
        self.assertEqual(metrics.saved_average_accuracy[0], 0.8)
        self.assertEqual(metrics.saved_fbeta_score[0], 0)
        self.assertEqual(metrics.saved_accuracy[0], 0)
        self.assertGreater(metrics.saved_loss[0], 0)


if __name__ == '__main__':
    # -- Unittests -- #
    unittest.main()
