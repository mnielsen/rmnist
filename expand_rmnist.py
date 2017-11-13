"""expand_rmnist.py
~~~~~~~~~~~~~~~~~~~

Take the RMNIST training images, and create expanded sets of training
images, by displacing each training image in a 5 x 5 region.  Save the
resulting file to ../data/rmnist_expanded_n.pkl.gz.

"""

from __future__ import print_function

#### Libraries

# Standard library
import cPickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np

print("Expanding the RMNIST training sets")

sizes = [1, 5, 10]

def shift(image, d, axis):
    if d == 0: return image
    if d == 1: index = 0
    if d == -1: index = 27
    new_img = np.roll(image, d, axis)
    if axis == 0:
        if d == 2: new_img[1, :] = np.zeros(28)
        if d >= 1: new_img[0, :] = np.zeros(28)
        if d <= -1: new_img[27, :] = np.zeros(28)
        if d == -2: new_img[26, :] = np.zeros(28)
    if axis == 1:
        if d == 2: new_img[:, 1] = np.zeros(28)
        if d >= 1: new_img[:, 0] = np.zeros(28)
        if d <= -1: new_img[:, 27] = np.zeros(28)
        if d == -2: new_img[:, 26] = np.zeros(28)
    return new_img

for n in sizes:
    print("\n\nExpanding RMNIST/{}".format(n))
    #if os.path.exists("data/rmnist_expanded_{}.pkl.gz".format(n)):
    #    print("The expanded training set already exists.  Exiting.")
    #    continue
    f = gzip.open("data/rmnist_{}.pkl.gz".format(n), 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    expanded_training_pairs = []
    j = 0 # counter
    for x, y in zip(training_data[0], training_data[1]):
        j += 1
        if j % 10 == 0: print("Expanding image number", j)
        image = np.reshape(x, (28, 28))
        for dx in [2, 1, 0, -1, 2]:
            for dy in [2, 1, 0, -1, 2]:
                expanded_training_pairs.append(
                    (np.reshape(shift(shift(image, dx, 0), dy, 1), 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving expanded data.")
    f = gzip.open("data/rmnist_expanded_{}.pkl.gz".format(n), "w")
    cPickle.dump((expanded_training_data, validation_data, test_data), f)
    f.close()





