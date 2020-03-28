"""
This code plots an array image and indicates the label with the index of the image.
"""
import matplotlib.pyplot as plt
import numpy.random as rd


def show_image(image, index, label, real_label):
    """
    Plots the image and indicate the index and the label.

    Args:
        image (array, any shape with at least 2 dimensions): the image.

        index (int): the index of the image.

        label (int): the estimated label of the image.

        real_label (int): the true label of the image.
    """
    plt.imshow(image)
    plt.title('Image {}, Estimated label : {}, True label : {}'.format(index, label, real_label))
    plt.show()


if __name__ == "__main__":
    IMAGE = rd.randint(0, 255, (32, 32, 3))
    show_image(IMAGE, 90, 1, 2)
