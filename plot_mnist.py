"""plot_mnist.py
~~~~~~~~~~~~~~~~

Use to plot MNIST images.
"""

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_mnist(elts, m, n):
    """Plot MNIST images in an m by n table. Note that we crop the images
    so that they appear reasonably close together.  Note that we are
    passed raw MNIST data and it is reshaped.

    """
    fig = plt.figure()
    images = [elt.reshape(28, 28) for elt in elts]
    img = np.concatenate([np.concatenate([images[m*y+x] for x in range(m)], axis=1)
                          for y in range(n)], axis=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(img, cmap = matplotlib.cm.binary)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()
