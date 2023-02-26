from collections import Counter
from scipy.special import bernoulli, gammaln
from scipy.interpolate import interp1d
import numpy as np
from .inverse_coefs import _Afuncs


#######################
# PARTITIONS
#######################

def _accel_asc(n):
    """
    Fast method for getting all partitions of the integer n, [0, 1].

    [0] https://jeromekelleher.net/generating-integer-partitions.html
    [1] https://arxiv.org/abs/0909.2331
    """
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def _partitions(n):
    """
    Call the _accel_asc method to get the parititions of n.  Each of partition
    will be a list.  E.g., for the number 4, one of these will be [1, 1, 2].
    We apply Counter to this, to turn it into the dict {1: 2, 2:1}, and return
    the result -- a list of these dict form partitions.
    """
    res = _accel_asc(n)
    return [Counter(part) for part in res]


#######################
# NEWTON HOP
#######################

def _higher_derivative_chain_rule(f_list, g_list):
    """
    Return the n-th derivative of f(g(x)), given the first n derivatives of f
    wrt to x at g(x) and the first n derivatives of g(x) wrt x at x.

    For this we use the Faa di Bruno theorem -- see (2) of [0].

    [0] https://hoyleanalytics.org/2017/05/20/

    parameters
    ----------
    f_list: list
        list of first n derivatives of f

    g_list: list
        list of first n derivatives of g
    """
    # preliminary checks on derivative list length agreement
    n = len(f_list)
    msg = "Chain rule: f_list and g_list do not agree in length"
    assert len(g_list) == n, msg

    # step 2: get partitions of n
    partitions = _partitions(n)

    # step 3: loop through partitions
    res = 0.0
    for part_dict in partitions:
        # loop through the terms depending on g's derivatives
        m_j_sum = 0
        term = 0.0
        sign = 1
        for j in part_dict:
            # prep and factorial in front of product
            m_j = part_dict[j]
            m_j_sum += m_j
            term -= gammaln(m_j + 1)

            # get needed derivative of g and track sign contribution
            g_j = g_list[j - 1]

            # exit early if g_j = 0
            if g_j == 0:
                term = -np.inf
                break

            # else, tack on the contribution to the log of the term
            sign *= np.sign(g_j) ** m_j
            term += m_j * (np.log(np.abs(g_j)) - gammaln(j + 1))

        # appropriate derivative of f, its sign, and the n! term.  if zero,
        # continue here to avoid warning on log(0) below
        f_m_j_sum = f_list[m_j_sum - 1]

        if f_m_j_sum == 0:
            continue
        
        sign *= np.sign(f_m_j_sum)
        term += np.log(np.abs(f_m_j_sum))
        term += gammaln(n + 1)

        res += np.exp(term) * sign

    return res


def _final_newton_hop(
        x_vals, y0, y_prime_vals, left_derivative_vals, right_derivative_vals,
        return_y=False
    ):
    """
    Take a final Newton, first order hop to further reduce error after the
    inchworm process is complete.  We assume that we can't evaluate the y
    function directly, so to carry this out we need to esimate the final y
    position, given the derivative information we have available.  To do this,
    we use the Euler-Maclaurin (EM) sum formula [0], applied to the integral of
    y'. This gives us an estimate for our final y value.  From there, we Newton
    hop to reduce the y value closer to 0 still.

    To apply the EM formula to a given integral, the integrand needs to be
    sampled at a set of evenly spaced points.  The samples we have available
    are the positions the inchworm has visited on its way to the root -- and
    To get around this issue, we first generate a spline to map the values

         0, 1, 2, .., n --> x_vals[0], ..., x_vals[n]

    The integral of y' is then equal to the following integral

        int_0^n f'(x(x_tilde)) (d x / d x_tilde) d x_tilde

    We can now estimate this latter integral via the EM formula.  This gives us
    a way to estimate the final y.  After we have this result, we Newon hop and
    return the updated x position.

    NOTE: We require the derivative values used here to be pre-computed.  This
    does not speed up the runtime very much for quick-to-evaluate functions.
    However, it could be useful for applications where the derivatives are also
    expensive to calculate.  In the simple alternative to this method just
    below, we assume the derivatives are easy to evaluate, and so just resample
    at even spacing -- much more convenient.

    [0] https://en.wikipedia.org/wiki/Euler%E2%80%93Maclaurin_formula

    parameters
    ----------
    x_vals : list
        The points where have sampled the derivative of y.

    y0 : float
        The initial value of y

    y_prime_vals : list
        The first derivative values, evaluated at each of the x_vals.

    left_derivative_vals : list
        The first m derivatives of y, evaluated at x_vals[0]

    right_derivative_vals : list
        The first m derivatives of y, evaluated at x_vals[-1]

    return_y : bool
        If True, we return the final y estimate instead of the final x value.
        This option is present for testing purposes only.

    NOTE:  In order to apply a k-th order spline here, we need at least k + 2
    terms.  To avoid an error, we set derivative_count here to the minimum of
    the number of samples and the length of the d_list.
    """
    # spline needs continuous dc - 1 partials --> use dc + 1.  spline algo we
    # use only accepts odd orders, so use 2 [dc // 2] + 1 below.  Finally, we
    # need at least a cubic spline, so threshold below at 3.
    x_vals = np.array(x_vals)
    x_tilde = np.arange(0, len(x_vals))

    derivative_count = min(len(left_derivative_vals), len(x_vals) - 2)
    spline_kind = max(2 * (derivative_count // 2) + 1, 3)

    spline = interp1d(x_tilde, x_vals, kind=spline_kind)

    # first term in Euler-Maclaurin (special case using all sample points)
    spline_derivatives = spline._spline.derivative(nu=1)(x_tilde).ravel()

    y_final = y0 * 1.0
    y_final += np.dot(y_prime_vals, spline_derivatives)
    y_final -= 0.5 * (y_prime_vals[0] * spline_derivatives[0]
        + y_prime_vals[-1] * spline_derivatives[-1]
    )

    # higher order EM terms
    for k in np.arange(1, derivative_count // 2 + 1):
        prefactor = bernoulli(2 * k)[-1] / np.math.factorial(2 * k)

        # left side -- get first 2k derivatives of our function and the spline
        spline_derivatives = [
            spline._spline.derivative(nu=i + 1)(x_tilde[0])[0]
                for i in range(2 * k)
        ]
        d_factor = _higher_derivative_chain_rule(
            left_derivative_vals[:2 * k], spline_derivatives
        )

        # right side -- get first 2k derivatives of our function and the spline
        spline_derivatives = [
            spline._spline.derivative(nu=i + 1)(x_tilde[-1])[0]
                for i in range(2 * k)
        ]
        d_factor -= _higher_derivative_chain_rule(
            right_derivative_vals[:2 * k], spline_derivatives
        )

        # add this term to the EM estimate
        y_final += prefactor * d_factor

    # for tests -- if return_y, return the y_final value instead
    if return_y:
        return y_final

    x_val = x_vals[-1] - y_final / y_prime_vals[-1]

    return x_val


def _final_newton_hop_simple(x_vals, y0, derivatives, return_y=False):
    """
    Take a final Newton, first order hop to further reduce error after the
    inchworm process is complete.  We assume that we can't evaluate the y
    function directly, so to carry this out we need to esimate the final y
    position, given the derivative information we have available.  To do this,
    we use the Euler-Maclaurin (EM) sum formula [0], applied to the integral of
    y'. This gives us an estimate for our final y value.  From there, we Newton
    hop to reduce the y value closer to 0 still.

    In this method, we assume that the x_vals are equally spaced.  In this
    situation, we do not need to develop a spline and can apply the EM formula
    directly to the passed samples.

    [0] https://en.wikipedia.org/wiki/Euler%E2%80%93Maclaurin_formula

    parameters
    ----------
    x_vals : list
        The points where have sampled the derivative of y.

    y0 : float
        The initial value of y

    derivatives : list
        list holding derivative functions of y.

    return_y : bool
        If True, we return the final y estimate instead of the final x value.
        This option is present for testing purposes only.
    """
    # prep
    delta_x = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1.0)
    derivative_count = min(len(derivatives), len(x_vals) - 2)

    # first term in Euler-Maclaurin (special case using all sample points)
    y_prime_vals_scaled = derivatives[0](x_vals) * delta_x

    y_final = y0 * 1.0
    y_final += np.sum(y_prime_vals_scaled)
    y_final -= 0.5 * (y_prime_vals_scaled[0] + y_prime_vals_scaled[-1])

    # higher order EM terms
    for k in np.arange(1, derivative_count // 2 + 1):
        prefactor = bernoulli(2 * k)[-1] / np.math.factorial(2 * k)
        y_final += prefactor * delta_x ** (2 * k) * (
            + derivatives[2 * k - 1](x_vals[0])
            - derivatives[2 * k - 1](x_vals[-1])
        )

    # for tests -- if return_y, return the y_final value instead
    if return_y:
        return y_final

    x_val = x_vals[-1] - y_final / derivatives[0](x_vals[-1])

    return x_val


#######################
# INVERSE COEFFICIENTS
#######################

def _generate_nth_inverse_coefficient_terms(n):
    """
    Generate the n-th coefficient in the series inversion formula. That is, all
    the information in the inverse formula that does not depend on the a_i.

    format for the i-th term returned:
    ----------------------------------
        [sign, log coef, part_dict]
        - sign is the sign of term, either 1 or -1

        - log coef is the natural log of the factorial terms preceeding the
          factors dependent on the ai

        - part_dict specifies which terms appear in the the partition and their
          counts.  These determine which ai appear in the term and the power
          they are raised to.
 
    parameters
    ----------
    n : int
        The coefficient index, starting with 1.

    NOTE:  The n=1 term should not call this method as the sum does not apply
    in this case.  In our code, we don't call for any n <= 7, but instead use
    hard-coded functions for these cases.
    """
    partitions = _partitions(n - 1)

    terms = []

    for part_dict in partitions:
        # construct the coefficient
        m_j_sum = 0
        term = 0.0
        for j in part_dict:
            # prep and factorial in front of product
            m_j = part_dict[j]
            m_j_sum += m_j
            term -= gammaln(m_j + 1)
        
        # now add in the numerator
        term += gammaln(n + m_j_sum) - gammaln(n)
        terms.append([(-1) ** m_j_sum, term, part_dict])
        
    return terms


def _inverse_coefs(a, use_analytic_forms=True):
    """
    Get inverse coefficients, given coefficients of original power series. That
    is, a should hold the first n coefficients of the forward power series

            dy ~ a[0] dx + a[1] dx^2 + ...

    Given these, we return the first n coefficients A of the inverse power
    series,

            dx = A[0] dy + A[1] dy^2 + ...

    parameters
    ----------
    a : list
        list of forward power series coefficients.

    use_analytic_forms : bool
        If True, we use the hard-coded forms where available.  This is toggled
        to False in our tests of the method, where we check for agreement of
        the hard-coded and numerical, series approaches.

    notes
    ----- 
    For the first seven inverse coefficients, we have hard-coded the inverse
    coefficient formulas.  This helps avoid float round off error. eyond the
    seventh coefficient, we use the general series formula [1] to evaluate the
    terms.  In some simple tests, we have found that this formula generates
    round off error at the ~15th decimal place.

    [1] https://mathworld.wolfram.com/SeriesReversion.html
    """
    coef_count = len(a)
    coefs = np.zeros(coef_count)
    coefs[0] = 1.0 / a[0]

    if coef_count == 1:
        return coefs

    for n in np.arange(2, len(a) + 1):
        if (use_analytic_forms) and (n <= 8):
            coefs[n - 1] = _Afuncs[n - 1](a)
            continue

        terms = _generate_nth_inverse_coefficient_terms(n)
        coef = 0.0
        for term_info in terms:
            sign = term_info[0] * 1.0
            term = term_info[1] * 1.0
            part_dict = term_info[2]
            for j in part_dict:
                # exit early if the term is 0
                if a[j] == 0:
                    term = -np.inf
                    break

                # otherwise, tack on contribution to log of term
                sign *= (np.sign(a[j]) * np.sign(a[0])) ** part_dict[j]
                term += part_dict[j] * (
                    np.log(np.abs(a[j])) - np.log(np.abs(a[0]))
                )
            coef += np.exp(term) * sign
        coef = coef / (n * a[0] ** n)
        coefs[n - 1] = coef

    return coefs
