from unittest import TestCase
import numpy as np
import inchwormrf



class TestMethods():
    """
    Check agreement of methods in the inchworm file.
    """
    def test_inverse_coefficients_agreement(TestCase):
        """
        Check that the hard-coded and series forms give the same results for a
        few different choices of the "forward" coefficients.
        """
        a_options = [
            [1.0],
            [1.0, 2.0, 4.0],
            [1.0, -2.0, -4.0],
            [1.0, 2.0, 4.0, 5.0, 7.0, 10.0],
            [1.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 3.0, -1.0, 4.0, 0.0, 0.0, 0.0, 1.0],
        ]

        msg = "hard-coded and series inverse coefficients do not agree."

        for a in a_options:

            res1 = inchwormrf._inverse_coefs(a)
            res2 = inchwormrf._inverse_coefs(a, use_analytic_forms=False)

            assert np.all(np.isclose(res1, res2)), msg

    def test_final_y_estimate_improvement_with_d_count(TestCase):
        """
        The EM sum rule should give a more refined estimate for our final y
        value (after inching) as we pass more derivatives to it.
        """
        y_function = lambda x: x ** 10 - 1
        derivatives = [
            lambda x: 10 * x ** 9,
            lambda x: 10 * 9 * x ** 8,
            lambda x: 10 * 9 * 8 * x ** 7,
            lambda x: 10 * 9 * 8 * 7 * x ** 6,
            lambda x: 10 * 9 * 8 * 7 * 6 * x ** 5,
            lambda x: 10 * 9 * 8 * 7 * 6 * 5 * x ** 4,
            lambda x: 10 * 9 * 8 * 7 * 6 * 5 * 4 * x ** 3,
            lambda x: 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * x ** 2,
            lambda x: 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * x,
            lambda x: 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1,
        ]

        x_vals = np.linspace(2.0, 1.0 + 0.0001, 20)
        y0 = y_function(x_vals[0])

        y_prime_vals = derivatives[0](x_vals)

        true_y_final = y_function(x_vals[-1])

        y_final_estimates = []
        for i in np.arange(1, 11, 2):
            local_derivatives = derivatives[:i]
            left_derivative_vals = [d(x_vals[0]) for d in local_derivatives]
            right_derivative_vals = [d(x_vals[-1]) for d in local_derivatives]
            y_final_estimates.append(
                inchwormrf._final_newton_hop(
                    x_vals=x_vals, y0=y0, y_prime_vals=y_prime_vals,
                    left_derivative_vals=left_derivative_vals,
                    right_derivative_vals=right_derivative_vals, return_y=True,
                )
            )

        errors = np.abs(y_final_estimates - true_y_final)

        msg = "EM sum rule not always more accurate with increased d count"
        assert np.all(errors[1:] - errors[:-1] < 0), msg


    def test_root_error_improvement_with_step_count(TestCase):
        """
        Check better performance as N goes up.
        """
        y_function = lambda x: x ** 3 - 5
        derivatives = [
           lambda x: 3 * x ** 2,
        ]

        x0 = 2.0
        y0 = y_function(x0)

        root_estimates = []
        for N in [10, 100, 1000]:
            root = inchwormrf.inchworm(
                x0,
                y0,
                N,
                derivatives,
                newton_pass=False
            )
            root_estimates.append(root)

        errors = np.abs(np.array(root_estimates) - 5 ** (1/3.))

        msg = "Error does not always decrease with N"
        assert np.all(errors[1:] - errors[:-1] < 0), msg


    def test_root_estimate_improvement_with_derivative_count(TestCase):
        """
        Check we get improved estimates as the number of derivatives passed
        increases.
        """
        y_function = lambda x: x ** 3 - 5
        derivatives = [
            lambda x: 3 * x ** 2,
            lambda x: 3 * 2 * x,
            lambda x: 3 * 2 * 1,
        ]

        x0 = 2.0
        y0 = y_function(x0)
        N = 1000

        root_estimates = []
        for i in [1, 2, 3]:
            local_derivatives = derivatives[:i]
            root = inchwormrf.inchworm(
                x0,
                y0,
                N,
                local_derivatives,
                newton_pass=False
            )
            root_estimates.append(root)

        errors = np.abs(np.array(root_estimates) - 5 ** (1/3.))

        msg = "Error does not always decrease with N"
        assert np.all(errors[1:] - errors[:-1] < 0), msg


    def test_higher_derivative_chain_rule(TestCase):
        """
        Check our higher derivative chain rule function is giving correct
        results.

        We check that method gives correct derivatives for sin(x)^2. Here,

            f = x^2, g = sin(x)

        We need to evaluate derivatives of f at x = sin(x_tilde), and those of
        g at x_tilde.
        """
        x_tilde = 0.25
        x = np.sin(x_tilde)
        msg = 'higher derivative chain rule disagreement %2.10f, %2.10f'

        # FIRST DERIVATIVE
        f_list = [2 * x]
        g_list = [np.cos(x_tilde)]

        exact = 2 * np.sin(x_tilde) * np.cos(x_tilde)
        chain_rule = inchwormrf._higher_derivative_chain_rule(f_list, g_list)
        assert np.isclose(exact, chain_rule, rtol=10 ** -6, atol=0), msg % (
            exact, chain_rule
        )

        # SECOND DERIVATIVE
        f_list = [2 * x, 2]
        g_list = [np.cos(x_tilde), -np.sin(x_tilde)]

        exact = 2 * np.cos(x_tilde) ** 2 - 2 * np.sin(x_tilde) ** 2
        chain_rule = inchwormrf._higher_derivative_chain_rule(f_list, g_list)
        assert np.isclose(exact, chain_rule, rtol=10 ** -6, atol=0), msg

        # THIRD DERIVATIVE
        f_list = [2 * x, 2, 0]
        g_list = [np.cos(x_tilde), -np.sin(x_tilde), -np.cos(x_tilde)]

        exact = - 8 * np.cos(x_tilde) * np.sin(x_tilde)
        chain_rule = inchwormrf._higher_derivative_chain_rule(f_list, g_list)
        assert np.isclose(exact, chain_rule, rtol=10 ** -6, atol=0), msg


        # FOURTH DERIVATIVE
        f_list = [2 * x, 2, 0, 0]
        g_list = [
            np.cos(x_tilde),
            -np.sin(x_tilde),
            -np.cos(x_tilde),
            np.sin(x_tilde),
        ]

        exact = - 8 * np.cos(x_tilde) ** 2 + 8 * np.sin(x_tilde) ** 2
        chain_rule = inchwormrf._higher_derivative_chain_rule(f_list, g_list)
        assert np.isclose(exact, chain_rule, rtol=10 ** -6, atol=0), msg

        # FIFTH DERIVATIVE
        f_list = [2 * x, 2, 0, 0, 0]
        g_list = [
            np.cos(x_tilde),
            -np.sin(x_tilde),
            -np.cos(x_tilde),
            np.sin(x_tilde),
            np.cos(x_tilde)
        ]

        exact = 32 * np.sin(x_tilde) * np.cos(x_tilde)
        chain_rule = inchwormrf._higher_derivative_chain_rule(f_list, g_list)
        assert np.isclose(exact, chain_rule, rtol=10 ** -6, atol=0), msg

    def test_hop_improvement(TestCase):
        """
        Check that taking a final Newton hop improves root estimate.
        """
        # goal: find fifth root of 3
        y_function = lambda x: x ** 5 - 3

        # derivatives -- note: accuracy improves w/ derivative count supplied
        derivatives = [
            lambda x: 5 * x ** 4,                  # 1st derivative of y
            lambda x: 20 * x ** 3,                 # 2nd ...
            lambda x: 60 * x ** 2,                 # 3rd ...
            lambda x: 120 * x ** 1,                # 4th ...
        ]
        x0 = 2.0
        y0 = y_function(x0)
        true_root = 3 ** 0.2

        inch_root = inchwormrf.inchworm(
            x0=x0,
            y0=y0,
            N=10**3,
            derivatives=derivatives,
            newton_pass=False
        )

        inch_with_newton_root = inchwormrf.inchworm(
            x0=x0,
            y0=y0,
            N=10**3,
            derivatives=derivatives,
            newton_pass=True
        )

        inch_error = np.abs(inch_root - true_root)
        inch_with_newton_error = np.abs(inch_with_newton_root - true_root)

        msg = "Newton hop did not improve root estimate"
        assert inch_with_newton_error < inch_error, msg

    def test_general_and_hard_coded_inchworms(TestCase):
        """
        Test that our hard-coded methods agree with the general code.
        """
        y_function = lambda x: np.sin(x)
        derivatives = [
            lambda x: np.cos(x),
            lambda x: -np.sin(x),
            lambda x: -np.cos(x),
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: -np.sin(x),
            lambda x: -np.cos(x),
            lambda x: np.sin(x),
        ]

        x0 = 0.5
        y0 = y_function(x0)
        N = 10 ** 2
        true_root = 0.0

        # 1-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:1],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm1(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch1 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:1],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm1newton(
            x0,
            y0,
            N=N,
	    derivative1=lambda x: np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch1newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg


        # 2-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:2],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm2(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch2 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:2],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm2newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch2newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # 3-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:3],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm3(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch3 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:3],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm3newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch3newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg


        # 4-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:4],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm4(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch4 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:4],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm4newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch4newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # 5-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:5],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm5(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch5 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:5],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm5newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch5newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # 6-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:6],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm6(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
            derivative6=lambda x: -np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch6 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:6],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm6newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
            derivative6=lambda x: -np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch6newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # 7-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:7],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm7(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
            derivative6=lambda x: -np.sin(x),
            derivative7=lambda x: -np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch7 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:7],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm7newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
            derivative6=lambda x: -np.sin(x),
            derivative7=lambda x: -np.cos(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch7newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # 8-DERIVATIVE METHODS
        # case 1: no newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:8],
            newton_pass=False,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm8(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
            derivative6=lambda x: -np.sin(x),
            derivative7=lambda x: -np.cos(x),
            derivative8=lambda x: np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch8 don't agree: %2.8f, %2.8f" %(error_gen, error_hc)
        assert np.isclose(error_gen, error_hc, atol=1.5, rtol=0), msg

        # case 2: yes newton
        root_gen = inchwormrf.inchworm(
            x0,
            y0,
            N=N,
            derivatives=derivatives[:8],
            newton_pass=True,
            farm_out=False,
        )
        error_gen = np.log(np.abs(root_gen - true_root))

        root_hc = inchwormrf.inchworm8newton(
            x0,
            y0,
            N=N,
            derivative1=lambda x: np.cos(x),
            derivative2=lambda x: -np.sin(x),
            derivative3=lambda x: -np.cos(x),
            derivative4=lambda x: np.sin(x),
            derivative5=lambda x: np.cos(x),
            derivative6=lambda x: -np.sin(x),
            derivative7=lambda x: -np.cos(x),
            derivative8=lambda x: np.sin(x),
        )
        error_hc = np.log(np.abs(root_hc - true_root))
        msg = "inch and inch8newton don't agree: %2.8f, %2.8f" % (
            error_gen, error_hc
        )
        assert np.isclose(error_gen, error_hc, atol=2.5, rtol=0), msg
