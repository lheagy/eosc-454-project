import numpy as np
from linear_problem import kernel_function

n_cells = 100
n_nodes = n_cells + 1
x_nodes = np.linspace(0, 1, n_nodes)

n_kernels = 5

p = -0.05
q = 0.1
j = 1

def test_kernel_size():
    output = kernel_function(x_nodes, j, p, q)
    assert len(output) == len(x_nodes)

def test_kernel_no_decay():
    test_cosine = kernel_function(x_nodes, j, 0, q)
    analytic_cosine = np.cos(2 * np.pi * j * q * x_nodes)
    print(
        f"norm numeric: {np.linalg.norm(test_cosine):1.2e}, norm analytic: {np.linalg.norm(analytic_cosine):1.2e}"
    )
    assert np.allclose(test_cosine, analytic_cosine)

def test_p_is_positive():
    test_p_positive = kernel_function(x_nodes, j, p*-1, q)
