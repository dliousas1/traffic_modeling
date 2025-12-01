import pytest
import numpy as np

from evaluate_f import Parameters, eval_f
from provided_solvers.SimpleSolver import SimpleSolver
from evaluate_Jf import eval_Jf_analytic_linear

@pytest.mark.parametrize("x, p, expected_f", [
    # Test case 1: One car
    (
        [0.0, 1.0],
        [Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
        [1.0, 0.0]
    ),
    # Test case 2: Two cars, first car accelerating
    (
        [0.0, 1.0, 10.0, 0.0],
        [Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0), Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
        [1.0, 1.0/1.0 * (10.0 - 0.0) + 0.5 * (0.0 - 1.0) - 1.0*1.0, 0.0, 0.0]
    ),
    # Test case 3: Two cars, first car decelerating
    (
        [9.0, 4.0, 10.0, 5.0],
        [Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0), Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
        [4.0, 1.0/1.0 * (10.0 - 9.0) + 0.5 * (5.0 - 4.0) - 1.0*4.0, 5.0, 0.0]
    ),
    # Test case 4: Three cars with different states and parameters
    (
        [0.0, 1.0, 10.0, 2.0, 20.0, 3.0],
        [Parameters(1.5, 0.75, 1.2, 0.0, 0.0, 0.0), Parameters(0.8, 0.4, 0.9, 0.0, 0.0, 0.0), Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
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
        [Parameters(12.5, 0.8, 5.0, 0.0, 0.0, 0.0), Parameters(1.2, 1.5, 3.0, 0.0, 0.0, 0.0), Parameters(5.3, 2.4, 1.5, 0.0, 0.0, 0.0)],
        [
            5.0, 0.0, 5.0, 0.0, 5.0, 0.0
        ]
    ),

    # Test case 6: Two cars with zero velocities
    (
        [0.0, 0.0, 10.0, 0.0],
        [Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0), Parameters(1.0, 0.5, 1.0, 0.0, 0.0, 0.0)],
        [0.0, 10.0, 0.0, 0.0]
    ),
])
def test_evaluate_f_linear(x, p, expected_f):
    """
    Test that eval_f computes the correct dynamics function when the system is linear 
    (i.e. the coefficients in the exponential braking term are 0.0).
    """
    param_dict = {"parameters": p, "dxFD":1e-8}
    # x_vector = np.array([item for state in x for item in (state.position, state.velocity)])
    f = eval_f(x, param_dict)
    assert np.allclose(f, expected_f), f"Expected {expected_f}, but got {f}"

    # Also test using stamped dynamics
    A = eval_Jf_analytic_linear(x, param_dict, None)
    f_stamped = A @ x
    assert np.allclose(f, f_stamped), f"Stamped dynamics {f_stamped} do not match eval_f {f}"


@pytest.mark.parametrize("x, p, expected_f", [
    # Test case 1: One car with nonlinear braking (should behave the same as linear since only one car)
    (
        [0.0, 1.0],
        [Parameters(1.0, 0.5, 1.0, 0.1, 0.2, 0.3)],
        [1.0, 0.0]
    ),
    # Test case 2: Two cars with nonlinear braking
    (
        [0.0, 1.0, 10.0, 0.0],
        [Parameters(1.0, 0.5, 1.0, 0.1, 0.2, 0.3), Parameters(1.0, 0.5, 1.0, 0.1, 0.2, 0.3)],
        [1.0,
         1.0/1.0 *(10.0 - 0.0 - 0.1 * np.exp(-0.2 * (10.0 - 0.0 - 0.3))) + 0.5 * (0.0 - 1.0) - 1.0*1.0,
         0.0,
         0.0]
    ),
])
def test_evaluate_f_nonlinear(x, p, expected_f):
    """
    Test that eval_f computes the correct dynamics function when the system is nonlinear 
    (i.e. the coefficients in the exponential braking term are non-zero).
    """
    param_dict = {"parameters": p, "dxFD":1e-8}
    f = eval_f(x, param_dict)
    assert np.allclose(f, expected_f), f"Expected {expected_f}, but got {f}"

@pytest.mark.parametrize("p", [
    # Test case 1: One car
    [Parameters(12.5, 0.8, 5.0, 0.0, 0.0, 0.0), Parameters(1.2, 1.5, 3.0, 0.0, 0.0, 0.0), Parameters(5.3, 2.4, 1.5, 0.0, 0.0, 0.0)]
])
def test_evaluate_f_steady_state(p):
    """
    Test that eval_f after running for many iterations results in the position and velocity differences between successive cars is equal to the expected analytical steady state value.
    """
    param_dict = {"parameters": p, "dxFD":1e-8}
    # Set arbitrary initial conditions (all zeros)
    x0 = np.zeros(2 * len(p))
    # Set leading car's initial velocity
    x0[-1] = 1.0  # Initial velocity of the leading car
    x_ss, t_ss = SimpleSolver(eval_f, x0, param_dict, lambda t: 0, 10000, 0.01, visualize=False)

    # Calculate the position and velocity differences between successive cars at the final time step
    position_differences = [x_ss[2*i+2, -1] - x_ss[2*i, -1] for i in range(len(p)-1)]
    velocity_differences = [x_ss[2*i+3, -1] - x_ss[2*i+1, -1] for i in range(len(p)-1)]

    # Calculate the expected steady state values
    expected_position_differences = [x_ss[2*i+1, -1] * p[i].tau for i in range(len(p)-1)]
    expected_velocity_differences = [0.0 for _ in range(len(p)-1)]

    # Assert that the computed differences are close to the expected values
    assert np.allclose(position_differences, expected_position_differences, atol=1e-5), f"Position differences {position_differences} do not match expected {expected_position_differences}"
    assert np.allclose(velocity_differences, expected_velocity_differences, atol=1e-5), f"Velocity differences {velocity_differences} do not match expected {expected_velocity_differences}"
