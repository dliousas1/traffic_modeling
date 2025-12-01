import numpy as np
import pytest
from evaluate_f import Parameters, eval_f
from evaluate_Jf import eval_Jf_analytic_linear
from provided_solvers.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference

@pytest.mark.parametrize("p, expected_Jf", [
    # Test case 1: One car
    (
        [Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
        np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    ),
    # Test case 2: Two cars
    (
        [Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0), Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
        np.array([[0.0, 1.0, 0.0, 0.0],
                  [ -1.0, -1.5, 1.0, 0.5],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]])
    ),
])
def test_evaluate_Jf_linear(p, expected_Jf):
    """
    Test that eval_Jf_analytic_linear and eval_Jf_FiniteDifference compute the 
    correct Jacobian when the system is linear (i.e. the coefficients in
    the exponential braking term are 0.0).
    """
    # Evaluate the Jacobian analytically
    Jf_analytic = eval_Jf_analytic_linear(None, {"parameters": p}, None)
    assert np.allclose(Jf_analytic, expected_Jf), f"Analytic Jacobian does not match expected value."

    # Use a sample state vector x0
    n = len(p)
    x0 = np.zeros(2 * n)

    param_dict = {"parameters": p, "dxFD":1e-8}
    # Evaluate the Jacobian using finite difference
    Jf_fd, dxFD = eval_Jf_FiniteDifference(eval_f, x0.reshape(-1, 1), param_dict, 0.0)
    assert np.allclose(Jf_fd, expected_Jf, atol=1e-5 )


@pytest.mark.parametrize("p, expected_Jf", [
    # Test case 1: One car with nonlinear terms
    (
        [Parameters(1.0, 0.5, 1.0, 0.1, 0.1, 0.1)],
        np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    ),
    # Test case 2: Two cars with nonlinear terms
    (
        [Parameters(1.0, 0.5, 1.0, 0.1, 0.1, 0.1), Parameters(1.0, 0.5, 1.0, 0.1, 0.1, 0.1)],
        np.array([[0.0, 1.0, 0.0, 0.0],
                  [-1.0101005, -1.5, 1.0101005, 0.5],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]])
    ),
])
def test_evaluate_Jf_nonlinear(p, expected_Jf):
    """
    Test that eval_Jf_analytic_linear matches the expected Jacobian when
    the system is nonlinear (i.e. the coefficients in the exponential braking 
    term are non-zero).
    """
    # Evaluate the Jacobian using finite difference
    n = len(p)
    x0 = np.zeros(2 * n)
    param_dict = {"parameters": p, "dxFD":1e-8}
    Jf_fd, dxFD = eval_Jf_FiniteDifference(eval_f, x0.reshape(-1, 1), param_dict, 0.0)
    assert np.allclose(Jf_fd, expected_Jf, atol=1e-5)