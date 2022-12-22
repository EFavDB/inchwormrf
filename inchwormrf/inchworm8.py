from scipy.interpolate import interp1d
import numpy as np
from .helpers import _final_newton_hop


def inchworm8(
        x0, y0, N, derivative1, derivative2, derivative3, derivative4,
        derivative5, derivative6, derivative7, derivative8,
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

    derivative7 : function
        function taking single argument, returns seventh derivative of target y
        function at x.

    derivative8 : function
        function taking single argument, returns eigth derivative of target y
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
    delta_y_7 = delta_y_1 ** 7
    delta_y_8 = delta_y_1 ** 8

    for _ in range(N):
        a0 = derivative1(x_val)
        a1 = derivative2(x_val) / 2.0
        a2 = derivative3(x_val) / 6.0
        a3 = derivative4(x_val) / 24.0
        a4 = derivative5(x_val) / 120.0
        a5 = derivative6(x_val) / 720.0
        a6 = derivative7(x_val) / 5040.0
        a7 = derivative8(x_val) / 40320.0

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
            + delta_y_7 / (a0 ** 13) * (
                + 132 * a1 ** 6
                - 330 * a0 * a1 ** 4 * a2 
                + 120 * a0 ** 2 * a1 ** 3 * a3
                - 36 * a0 ** 2 * a1 ** 2 * (-5 * a2 ** 2 + a0 * a4)
                + 4 * a0 ** 3 * (
                    -3 * a2 ** 3 + a0 * a3 ** 2 + 2 * a0 * a2 * a4
                )
                + 8 * a0 ** 3 * a1 * (-9 * a2 * a3 + a0 * a5)
                - a0 ** 5 * a6
            )
            + delta_y_8 / (a0 ** 15) * (
                - 429 * a1 ** 7
                + 1287 * a0 * a1 ** 5 * a2
                - 495 * a0 ** 2 * a1 ** 4 * a3
                + 165 * a0 ** 2 * a1 ** 3 * (-6 * a2 ** 2 + a0 * a4)
                - 45 * a0 ** 3 * a1 ** 2 * (-11 * a2 * a3 + a0 * a5)
                + 9 * a0 ** 4 * (
                    -5 * a2 ** 2 * a3 + a0 * a3 * a4 + a0 * a2 * a5
                )
                + 3 * a0 ** 3 * a1 * (
                    + 55 * a2 ** 3
                    - 30 * a0 * a2 * a4
                    + 3 * a0 * (-5 * a3 ** 2 + a0 * a6)
                )
                - a0 ** 6 * a7
            )
        )

    return x_val


def inchworm8newton(
        x0, y0, N, derivative1, derivative2, derivative3, derivative4,
        derivative5, derivative6, derivative7, derivative8,
    ):
    """
    Hard-coded inchworm with one derivative passed only, Newton hop.

    parameters
    ----------
    x0 : float
        initial guess for the root

    y0 : float
        initial value of the function

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

    derivative7 : function
        function taking single argument, returns seventh derivative of target y
        function at x.

    derivative8 : function
        function taking single argument, returns eigth derivative of target y
        function at x.

    N : int
        The number of inchworm steps to take in seraching for the root.  The
        estimate's accuracy will increase with this value, but the runtime
        also grows linearly in N.

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
    delta_y_7 = delta_y_1 ** 7
    delta_y_8 = delta_y_1 ** 8

    for index in range(N):
        a0 = derivative1(x_val)
        a1 = derivative2(x_val) / 2.0
        a2 = derivative3(x_val) / 6.0
        a3 = derivative4(x_val) / 24.0
        a4 = derivative5(x_val) / 120.0
        a5 = derivative6(x_val) / 720.0
        a6 = derivative7(x_val) / 5040.0
        a7 = derivative8(x_val) / 40320.0

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
            + delta_y_7 / (a0 ** 13) * (
                + 132 * a1 ** 6
                - 330 * a0 * a1 ** 4 * a2 
                + 120 * a0 ** 2 * a1 ** 3 * a3
                - 36 * a0 ** 2 * a1 ** 2 * (-5 * a2 ** 2 + a0 * a4)
                + 4 * a0 ** 3 * (
                    -3 * a2 ** 3 + a0 * a3 ** 2 + 2 * a0 * a2 * a4
                )
                + 8 * a0 ** 3 * a1 * (-9 * a2 * a3 + a0 * a5)
                - a0 ** 5 * a6
            )
            + delta_y_8 / (a0 ** 15) * (
                - 429 * a1 ** 7
                + 1287 * a0 * a1 ** 5 * a2
                - 495 * a0 ** 2 * a1 ** 4 * a3
                + 165 * a0 ** 2 * a1 ** 3 * (-6 * a2 ** 2 + a0 * a4)
                - 45 * a0 ** 3 * a1 ** 2 * (-11 * a2 * a3 + a0 * a5)
                + 9 * a0 ** 4 * (
                    -5 * a2 ** 2 * a3 + a0 * a3 * a4 + a0 * a2 * a5
                )
                + 3 * a0 ** 3 * a1 * (
                    + 55 * a2 ** 3
                    - 30 * a0 * a2 * a4
                    + 3 * a0 * (-5 * a3 ** 2 + a0 * a6)
                )
                - a0 ** 6 * a7
            )
        )
        x_vals[index + 1] = x_val

    # farm out to general newton hop method
    x_val = _final_newton_hop(
        x_vals, y0, [
            derivative1, derivative2, derivative3, derivative4,
            derivative5, derivative6, derivative7, derivative8,
        ]
    )

    return x_val
