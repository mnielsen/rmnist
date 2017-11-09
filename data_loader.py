"""data_loader
~~~~~~~~~~~~~~

A library to load the MNIST and RMNIST image data.

"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data(n=0, expanded=False):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    if n==0: name = "data/mnist.pkl.gz"
    else: name = "data/rmnist_"+str(n)+".pkl.gz"
    f = gzip.open(name, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def make_training_subset(n=10):
    """Make a subset of MNIST using n training examples of each digit and
    save into data/rmnist_n.pkl.gz, together with the complete
    validation and test sets.

    """ 
    td, vd, ts = load_data()
    import random
    random.seed(619) # use a standard seed to make this repeatable
    indices = range(50000)
    random.shuffle(indices)
    values = [(j, td[1][j]) for j in indices]
    indices_subset = [[v[0] for v in values if v[1] == j][:n] for j in range(10)]
    flattened_indices = [i for sub in indices_subset for i in sub]
    random.shuffle(flattened_indices)
    td0_prime = [td[0][j] for j in flattened_indices]
    td1_prime = [td[1][j] for j in flattened_indices]
    td_prime = (td0_prime, td1_prime)
    f = gzip.open('data/rmnist_'+str(n)+'.pkl.gz', 'wb')
    cPickle.dump((td_prime, vd, ts), f)
    f.close()






