import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def kernel_function(x_nodes, kernel_index, exponent, frequency):
    """
    Function for constructring decaying harmonic exponential functions

    Parameters
    ----------

    x_nodes: np.ndarray
        locations of the nodes of our 1D grid

    kernel_index: int, float
        parameter that chances periodicity and decay rate of kernel function

    exponent: float
        number in the exponent that controls the growth (for positive values) or decay (negative values) of our kernel function

    frequency: float
        oscillation rate of our kernel functions

    Returns
    -------

    np.ndarray : kernel function

    """
    return (
        np.exp(kernel_index * exponent * x_nodes)
        * np.cos(2 * np.pi * kernel_index * frequency * x_nodes)
    )
