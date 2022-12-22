from scipy.interpolate import interp1d
import numpy as np
from .helpers import _final_newton_hop


def inchworm2(x0, y0, N, derivative1, derivative2):
    """
    Hard-coded inchworm with two derivatives passed only, no Newton hop.

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

    derivative2 : function
        function taking single argument, returns second derivative of target y
        function at x.
    """
    # initial x
    x_val = x0 * 1.0

    # powers of delta y
    delta_y_1 = - y0 / float(N)
    delta_y_2 = delta_y_1 ** 2

    for _ in range(N):
        a0 = derivative1(x_val)
        a1 = derivative2(x_val) / 2.0
        x_val += delta_y_1 / a0 - delta_y_2 * (a1 / a0 ** 3)

    return x_val


def inchworm2newton(x0, y0, N, derivative1, derivative2):
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

    derivative2 : function
        function taking single argument, returns second derivative of target y
        function at x.
    """
    # initial x
    x_val = x0 * 1.0
    x_vals = np.zeros(N + 1)
    x_vals[0] = x_val

    # powers of delta y
    delta_y_1 = - y0 / float(N)
    delta_y_2 = delta_y_1 ** 2

    for index in range(N):
        a0 = derivative1(x_val)
        a1 = derivative2(x_val) / 2.0
        x_val += delta_y_1 / a0 - delta_y_2 * (a1 / a0 ** 3)
        x_vals[index + 1] = x_val

    # farm out to general newton hop method
    x_val = _final_newton_hop(
        x_vals, y0, [derivative1, derivative2]
    )

    return x_val
