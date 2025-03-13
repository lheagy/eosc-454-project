import numpy as np

def kernel_function(x, j, p, q):
    return (
        np.exp(j * p * x) *
        np.cos(2 * np.pi * j * q * x)
    )
