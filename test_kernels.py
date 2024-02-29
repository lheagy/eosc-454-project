import numpy as np
from linear_problem import kernel_function

x_nodes = np.linspace(0, 1, 100)

# case 1: frequency is zero
def test_decay():
    frequency = 0
    kernel_index = 1
    exponent = -1

    analytic = np.exp(exponent * kernel_index * x_nodes)
    numeric = kernel_function(
        x_nodes=x_nodes, kernel_index=kernel_index,
        exponent=exponent, frequency=frequency
    )
    print(
        np.linalg.norm(analytic), np.linalg.norm(numeric), np.linalg.norm(analytic-numeric)
    )
    assert np.allclose(analytic, numeric)
