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

    for index in range(N):
        x_val -= delta_y / derivative1(x_val)
        x_vals[index + 1] = x_val

    # farm out to general newton hop method
    x_val = _final_newton_hop(x_vals, y0, [derivative1])

    return x_val
