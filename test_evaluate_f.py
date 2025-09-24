import pytest
import numpy as np

from evaluate_f import Parameters, eval_f


@pytest.mark.parametrize("x, p, expected_f", [
    # Test case 1: One car
    (
        [0.0, 1.0],
        [Parameters(1.0, 0.5, 1.0)],
        [1.0, 0.0]
    ),
    # Test case 2: Two cars, first car accelerating
    (
        [0.0, 1.0, 10.0, 0.0],
        [Parameters(1.0, 0.5, 1.0), Parameters(1.0, 0.5, 1.0)],
        [1.0, 1.0/1.0 * (10.0 - 0.0) + 0.5 * (0.0 - 1.0) - 1.0*1.0, 0.0, 0.0]
    ),
    # Test case 3: Two cars, first car decelerating
    (
        [9.0, 4.0, 10.0, 5.0],
        [Parameters(1.0, 0.5, 1.0), Parameters(1.0, 0.5, 1.0)],
        [4.0, 1.0/1.0 * (10.0 - 9.0) + 0.5 * (5.0 - 4.0) - 1.0*4.0, 5.0, 0.0]
    ),
    # Test case 4: Three cars with different states and parameters
    (
        [0.0, 1.0, 10.0, 2.0, 20.0, 3.0],
        [Parameters(1.5, 0.75, 1.2), Parameters(0.8, 0.4, 0.9), Parameters(1.0, 0.5, 1.0)],
        [
            1.0,
            1.5/1.2 * (10.0 - 0.0) + 0.75 * (2.0 - 1.0) - 1.5*1.0,
            2.0,
            0.8/0.9 * (20.0 - 10.0) + 0.4 * (3.0 - 2.0) - 0.8*2.0,
            3.0,
            0.0,
        ]
    ),
    # Test case 5: Three cars spread such that acceleration is 0.0
    (
        [0.0, 5.0, 25.0, 5.0, 40.0, 5.0],
        [Parameters(12.5, 0.8, 5.0), Parameters(1.2, 1.5, 3.0), Parameters(5.3, 2.4, 1.5)],
        [
            5.0, 0.0, 5.0, 0.0, 5.0, 0.0
        ]
    ),
])
def test_evaluate_f(x, p, expected_f):
    """
    Test that eval_f computes the correct dynamics function for a set
    of test states and parameters.
    """
    # x_vector = np.array([item for state in x for item in (state.position, state.velocity)])
    f = eval_f(x, p)
    assert np.allclose(f, expected_f), f"Expected {expected_f}, but got {f}"

    # Also test using stamped dynamics
    from stamp_dynamics import stamp_dynamics
    A = stamp_dynamics(p)
    f_stamped = A @ x
    assert np.allclose(f, f_stamped), f"Stamped dynamics {f_stamped} do not match eval_f {f}"