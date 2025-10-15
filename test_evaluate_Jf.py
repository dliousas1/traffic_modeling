import numpy as np
import pytest
from evaluate_f import Parameters, eval_f
from evaluate_Jf import eval_Jf_analytic
from provided_solvers.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference

@pytest.mark.parametrize("p, expected_Jf", [
    # Test case 2: Two cars
    (
        [Parameters(1.0, 0.5, 1.0), Parameters(1.0, 0.5, 1.0)],
        np.array([[-1.5],])
    ),
    # Test case 3: Three cars with different parameters
    (
        [Parameters(1.0, 0.5, 1.0), Parameters(1.5, 0.3, 1.2), Parameters(0.8, 0.7, 0.9)],
        np.array([[-1.5, 1.0, 0.5],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.5/1.2, -1.5 - 0.3],])
    ),
])
def test_evaluate_Jf(p, expected_Jf):
    """
    Test that eval_Jf_analytic and eval_Jf_auto compute the correct Jacobian for a set of parameters.
    """
    # Evaluate the Jacobian analytically
    Jf_analytic = eval_Jf_analytic(p, order='alternate')
    assert np.allclose(Jf_analytic, expected_Jf), f"Analytic Jacobian does not match expected value."

    # Use a sample state vector x0
    n = len(p)
    x0 = np.zeros(2 * n)  # Arbitrary Sample state vector

    # Evaluate the Jacobian using finite difference
    Jf_fd, dxFD = eval_Jf_FiniteDifference(eval_f, x0.reshape(-1, 1), p, None)
    Jf_fd = Jf_fd[1:-2, 1:-2]  # Remove the first column/row and last two columns/rows
    assert np.allclose(Jf_fd, expected_Jf, atol=1e-5 )
