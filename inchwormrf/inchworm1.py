from scipy.interpolate import interp1d
import numpy as np
from .helpers import _final_newton_hop


def inchworm1(x0, y0, N, derivative1):
    """
    Hard-coded inchworm with one derivative passed only, no Newton hop.

    parameters
    ----------
    x0 : float
        initial guess for the root

    y0 : float
        initial value of the function

    N : int
        The number of inchworm steps to take in seraching for the root.

    derivative1 : function
        function taking single argument, returns first derivative of target y
        function at x.
    """
    x_val = x0 * 1.0
    delta_y = y0 / float(N)

    for _ in range(N):
        x_val -= delta_y / derivative1(x_val)

    return x_val

def inchworm1newton(x0, y0, N, derivative1):
    """
    Hard-coded inchworm with one derivative passed only, Newton hop.

    parameters
    ----------
    x0 : float
        initial guess for the root

    y0 : float
        initial value of the function

    N : int
        The number of inchworm steps to take in seraching for the root.

    derivative1 : function
        function taking single argument, returns first derivative of target y
        function at x.
    """
    # "hard-coded" inchworm process
    x_val = x0 * 1.0
    x_vals = np.zeros(N + 1)
    x_vals[0] = x_val

    delta_y = y0 / float(N)

    y_prime_vals = np.zeros(N + 1) 
    for index in range(N):
        a0 = derivative1(x_val)

        x_val -= delta_y / a0 
        x_vals[index + 1] = x_val

        # tracking derivatives for newton hop
        if index == 0:
            left_derivative_vals = [a0]
        y_prime_vals[index] = a0

    # final derivative vals we need to track
    y_prime_vals[-1] = derivative1(x_val)
    right_derivative_vals = [derivative1(x_val)]

    # farm out to general newton hop method
    x_val = _final_newton_hop(
        x_vals=x_vals, y0=y0, y_prime_vals=y_prime_vals,
        left_derivative_vals=left_derivative_vals,
        right_derivative_vals=right_derivative_vals,
    )

    return x_val
