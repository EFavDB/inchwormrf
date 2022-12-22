import numpy as np
from .helpers import _final_newton_hop, _final_newton_hop_simple

def newton(x0, y0, N, iterations, derivatives):
    """
    Return an estimate for a root of y, starting at x0.  Here, we repeatedly
    carry out approximate Newton hops based on our Euler-Maclaurin formula's
    estimate for the current function value.

    parameters
    ----------
    x0 : float
        initial guess for the root

    y0 : float
        initial value of the function

    N : int
        the number of inchworm steps to take in seraching for the root.

    iterations : int
        the number of Newton hops to take.

    derivatives : list
	list of derivative functions of y: The first derivative up to the k-th
        derivative.
    """
    # hop a first time
    x_val = x0 * 1.0
    y_prime = derivatives[0](x_val) 
    x_val -= y0 / y_prime

    # iteratively apply the approx _final_newton_hop and return
    for _ in range(iterations):
        x_vals = np.linspace(x0, x_val, N)
        x_val = _final_newton_hop_simple(x_vals, y0, derivatives)
 
    return x_val
