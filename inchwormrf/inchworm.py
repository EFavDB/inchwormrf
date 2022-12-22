import numpy as np
from .helpers import _inverse_coefs, _final_newton_hop
from .inchworm1 import inchworm1, inchworm1newton
from .inchworm2 import inchworm2, inchworm2newton
from .inchworm3 import inchworm3, inchworm3newton
from .inchworm4 import inchworm4, inchworm4newton
from .inchworm5 import inchworm5, inchworm5newton
from .inchworm6 import inchworm6, inchworm6newton
from .inchworm7 import inchworm7, inchworm7newton
from .inchworm8 import inchworm8, inchworm8newton

def inchworm(x0, y0, N, derivatives, newton_pass=False, farm_out=True):
    """
    Return an estimate for a root of y, starting at x0.

    parameters
    ----------
    x0 : float
        initial guess for the root

    y0 : float
        initial value of the function

    N : int
        The number of inchworm steps to take in seraching for the root.

    derivatives : list
	List of derivative functions of y: The first derivative up to the k-th
        derivative.

    newton_pass : bool
        If True, a final Newton pass is taken after the inchworm steps are
        completed.  This results in asymptotic convergence ~ N^{-2k}.  However,
        this is a computationally expensive step -- because of this, it is
        sometimes better to increase N with newton_pass=False.

    farm_out : bool
        If True, use one of the hard-coded implementations when available for
        the number of derivatives passed.  This will give a significant speed
        up when the option is available.
    """
    # setup
    d_count = len(derivatives)

    # farm out to a hard-coded method, if requsted and available
    if (farm_out) and (d_count < 9):
        return _farmer(x0, y0, N, derivatives, newton_pass, d_count)

    # run general inchworm code otherwise
    delta_y = - y0 / float(N)
    delta_y_value_powers = np.power(delta_y, np.arange(1, d_count + 1))
    factorial_factors = np.array([
        1.0 / np.math.gamma(order + 2) for order in range(d_count)
    ])

    # inch along
    x_vals = np.zeros(N +1)
    x_vals[0] = x0
    x_val = x0 * 1.0

    y_prime_coefs = np.zeros(d_count)
    for index in range(N):
        # STEP 1: get dy expansion in dx coefficients
        for idx in range(d_count):
            y_prime_coefs[idx] = (
                derivatives[idx](x_val) * factorial_factors[idx]
            )

        ## STEP 2: invert to get dx coefficients in dy
        inverse_coefs = _inverse_coefs(y_prime_coefs)

        ## STEP 3: dot inverse coefs into powers of delta y, return
        x_val += np.dot(delta_y_value_powers, inverse_coefs)
        x_vals[index + 1] = x_val

    if newton_pass:
        x_val =  _final_newton_hop(x_vals, y0, derivatives)

    return x_val


def _farmer(x0, y0, N, derivatives, newton_pass, d_count):
    """
    Helper farms root finding out to hard-coded methods.

    For slightly faster performance, call the hard-coded methods directly.

    parameters
    ----------
    x0 : float
        initial guess for the root

    y0 : float
        initial value of the function

    N : int
        The number of inchworm steps to take in seraching for the root.

    derivatives : function
        Function that returns an ordered list of derivative values of y: the
        first derivative, up to the k-th derivative.

    newton_pass : bool
        If True, a final Newton pass is taken after the inchworm steps are
        completed.  This results in asymptotic convergence ~ N^{-2k}.  However,
        this is a computationally expensive step -- because of this, it is
        sometimes better to increase N with newton_pass=False.

    d_count : number of derivatives passed to inchworm method.
    """
    if d_count > 8:
        msg = 'inchworm:  hard-coded methods not available for > 8 derivatives'
        raise ValueError(msg)

    # d_count = 1
    if d_count == 1 and newton_pass:
        return inchworm1newton(x0, y0, N, *derivatives)
    if d_count == 1 and (not newton_pass):
        return inchworm1(x0, y0, N, *derivatives)

    # d_count = 2
    if d_count == 2 and newton_pass:
        return inchworm2newton(x0, y0, N, *derivatives)
    if d_count == 2 and (not newton_pass):
        return inchworm2(x0, y0, N, *derivatives)

    # d_count = 3
    if d_count == 3 and newton_pass:
        return inchworm3newton(x0, y0, N, *derivatives)
    if d_count == 3 and (not newton_pass):
        return inchworm3(x0, y0, N, *derivatives)

    # d_count = 4
    if d_count == 4 and newton_pass:
        return inchworm4newton(x0, y0, N, *derivatives)
    if d_count == 4 and (not newton_pass):
        return inchworm4(x0, y0, N, *derivatives)

    # d_count = 5
    if d_count == 5 and newton_pass:
        return inchworm5newton(x0, y0, N, *derivatives)
    if d_count == 5 and (not newton_pass):
        return inchworm5(x0, y0, N, *derivatives)

    # d_count = 6
    if d_count == 6 and newton_pass:
        return inchworm6newton(x0, y0, N, *derivatives)
    if d_count == 6 and (not newton_pass):
        return inchworm6(x0, y0, N, *derivatives)

    # d_count = 7
    if d_count == 7 and newton_pass:
        return inchworm7newton(x0, y0, N, *derivatives)
    if d_count == 7 and (not newton_pass):
        return inchworm7(x0, y0, N, *derivatives)

    # d_count = 8
    if d_count == 8 and newton_pass:
        return inchworm8newton(x0, y0, N, *derivatives)
    if d_count == 8 and (not newton_pass):
        return inchworm8(x0, y0, N, *derivatives)
