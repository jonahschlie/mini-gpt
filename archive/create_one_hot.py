import numpy as np

def one_hot(index, length):
    vector = np.zeros(length, dtype=int)
    vector[index] = 1
    return vector