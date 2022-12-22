from scipy.interpolate import interp1d
import numpy as np
from .helpers import _final_newton_hop


def inchworm6(
        x0, y0, N, derivative1, derivative2, derivative3, derivative4,
        derivative5, derivative6
    ):
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

    derivative3 : function
        function taking single argument, returns third derivative of target y
        function at x.

    derivative4 : function
        function taking single argument, returns fourth derivative of target y
        function at x.

    derivative5 : function
        function taking single argument, returns fifth derivative of target y
        function at x.

    derivative6 : function
        function taking single argument, returns sixth derivative of target y
        function at x.
    """
    # initial x
    x_val = x0 * 1.0

    # powers of delta y
    delta_y_1 = - y0 / float(N)
    delta_y_2 = delta_y_1 ** 2
    delta_y_3 = delta_y_1 ** 3
    delta_y_4 = delta_y_1 ** 4
    delta_y_5 = delta_y_1 ** 5
    delta_y_6 = delta_y_1 ** 6

    for _ in range(N):
        a0 = derivative1(x_val)
        a1 = derivative2(x_val) / 2.0
        a2 = derivative3(x_val) / 6.0
        a3 = derivative4(x_val) / 24.0
        a4 = derivative5(x_val) / 120.0
        a5 = derivative6(x_val) / 720.0

        x_val += (
            delta_y_1 / a0
            - delta_y_2 * (a1 / a0 ** 3)
            + delta_y_3 * (2 * a1 ** 2 - a0 * a2) / a0 ** 5
            - delta_y_4 / (a0 ** 7) * (
                  + 5 * a1 ** 3
                  - 5 * a0 * a1 * a2
                  + a0 ** 2  * a3
            )
            + delta_y_5 / (a0 ** 9) * (
                + 14 * a1 ** 4
                - 21 * a0 * a1 ** 2 * a2
                + 3 * a0 ** 2 * a2 ** 2
                + 6 * a0 ** 2 * a1 * a3
                - a0 ** 3 * a4
            )
            + delta_y_6 / (a0 ** 11) * (
                -42 * a1 ** 5
                + 84 * a0 * a1 ** 3 * a2
                - 28 * a0 ** 2 * a1 ** 2 * a3
                + 7 * a0 ** 2 * a1 * (-4 * a2 ** 2 + a0 * a4)
                + a0 ** 3 * (7 * a2 * a3 - a0 * a5)
            )
        )

    return x_val


def inchworm6newton(
        x0, y0, N, derivative1, derivative2, derivative3, derivative4,
        derivative5, derivative6,
    ):
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

    derivative3 : function
        function taking single argument, returns third derivative of target y
        function at x.

    derivative4 : function
        function taking single argument, returns fourth derivative of target y
        function at x.

    derivative5 : function
        function taking single argument, returns fifth derivative of target y
        function at x.

    derivative6 : function
        function taking single argument, returns sixth derivative of target y
        function at x.
    """
    # initial x
    x_val = x0 * 1.0
    x_vals = np.zeros(N + 1)
    x_vals[0] = x_val

    # powers of delta y
    delta_y_1 = - y0 / float(N)
    delta_y_2 = delta_y_1 ** 2
    delta_y_3 = delta_y_1 ** 3
    delta_y_4 = delta_y_1 ** 4
    delta_y_5 = delta_y_1 ** 5
    delta_y_6 = delta_y_1 ** 6

    for index in range(N):
        a0 = derivative1(x_val)
        a1 = derivative2(x_val) / 2.0
        a2 = derivative3(x_val) / 6.0
        a3 = derivative4(x_val) / 24.0
        a4 = derivative5(x_val) / 120.0
        a5 = derivative6(x_val) / 720.0

        x_val += (
            delta_y_1 / a0
            - delta_y_2 * (a1 / a0 ** 3)
            + delta_y_3 * (2 * a1 ** 2 - a0 * a2) / a0 ** 5
            - delta_y_4 / (a0 ** 7) * (
                  + 5 * a1 ** 3
                  - 5 * a0 * a1 * a2
                  + a0 ** 2  * a3
            )
            + delta_y_5 / (a0 ** 9) * (
                + 14 * a1 ** 4
                - 21 * a0 * a1 ** 2 * a2
                + 3 * a0 ** 2 * a2 ** 2
                + 6 * a0 ** 2 * a1 * a3
                - a0 ** 3 * a4
            )
            + delta_y_6 / (a0 ** 11) * (
                - 42 * a1 ** 5
                + 84 * a0 * a1 ** 3 * a2
                - 28 * a0 ** 2 * a1 ** 2 * a3
                + 7 * a0 ** 2 * a1 * (-4 * a2 ** 2 + a0 * a4)
                + a0 ** 3 * (7 * a2 * a3 - a0 * a5)
            )
        )
        x_vals[index + 1] = x_val

    # farm out to general newton hop method
    x_val = _final_newton_hop(
        x_vals, y0, [
            derivative1, derivative2, derivative3, derivative4,
            derivative5, derivative6,
        ]
    )

    return x_val
