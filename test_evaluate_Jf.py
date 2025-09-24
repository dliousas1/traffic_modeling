import numpy as np
import pytest
from evaluate_f import Parameters, eval_f
from evaluate_Jf import eval_Jf_analytic, eval_Jf_auto
from stamp_dynamics import stamp_dynamics

@pytest.mark.parametrize("p, expected_Jf", [
    # Test case 1: One car
    (
        [Parameters(1.0, 0.5, 1.0)],
        np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    ),
    # Test case 2: Two cars
    (
        [Parameters(1.0, 0.5, 1.0), Parameters(1.0, 0.5, 1.0)],
        np.array([[0.0, 1.0, 0.0, 0.0],
                  [ -1.0, -1.5, 1.0, 0.5],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]])
    ),
])
def test_evaluate_Jf(p, expected_Jf):
    """
    Test that eval_Jf_analytic and eval_Jf_auto compute the correct Jacobian for a set of parameters.
    """
    # Evaluate the Jacobian analytically
    Jf_analytic = eval_Jf_analytic(p)
    assert np.allclose(Jf_analytic, expected_Jf), f"Analytic Jacobian does not match expected value."
    # Evaluate the Jacobian using automatic differentiation
    # Use a sample state vector x0
    n = len(p)
    x0 = np.zeros(2 * n)  # Arbitrary Sample state vector
    Jf_auto = eval_Jf_auto(eval_f, x0, p)
    assert np.allclose(Jf_auto, expected_Jf), f"Auto-diff Jacobian does not match expected value."
   


    