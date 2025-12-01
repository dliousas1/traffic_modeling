import numpy as np
from typing import Literal

from evaluate_f import Parameters


def eval_Jf_analytic_linear(x, p, u, order: Literal['alternate', 'position_first', 'velocity_first']='alternate') -> np.ndarray:
    """
    evaluates the Jacobian analytically for the linear portion of the dynamics

    Inputs:
    p: list of length (n,) Parameters objects, parameters vector.

    Outputs:
    Jf: numpy array of shape (2n, 2n), Jacobian of the linear dynamics.
    """
    Jf = stamp_dynamics_linear(p['parameters'], order=order)
    return Jf

def eval_Jf_analytic_nonlinear(x, p, u) -> np.ndarray:
    """
    evaluates the Jacobian analytically for the nonlinear portion of the dynamics

    Inputs:
    x: numpy array of shape (2n,), current state vector.
    p: list of length (n,) Parameters objects, parameters vector.
    u: unused for this system, included for compatibility.

    Outputs:
    Jf: numpy array of shape (2n, 2n), Jacobian of the nonlinear dynamics at state x.
    """
    Jf = stamp_dynamics_nonlinear(x, p['parameters'])
    return Jf

def eval_Jf_analytic_total(x, p, u) -> np.ndarray:
    """
    evaluates the Jacobian analytically for the total dynamics (linear + nonlinear)
    """
    Jf_linear = eval_Jf_analytic_linear(x, p, u)
    Jf_nonlinear = eval_Jf_analytic_nonlinear(x, p, u)
    Jf_total = Jf_linear + Jf_nonlinear
    return Jf_total

def stamp_dynamics_linear(
    p: list[Parameters], 
    order: Literal['alternate', 'position_first', 'velocity_first']='alternate'
) -> np.ndarray:
    """
    Stamps the dynamics of a system of vehicles into the global system matrices A and b.
    
    Inputs:
    p: list of length (n,) Parameters objects, parameters vector.
    
    Outputs:
    A: numpy array of shape (2n, 2n), system dynamics matrix.
    """

    n = len(p)
    A = np.zeros((2*n, 2*n))

    for i in range(n):
        alpha_i, beta_i, tau_i = p[i].alpha, p[i].beta, p[i].tau

        if order == 'alternate':
            pos_index_this_car = 2 * i
            vel_index_this_car = 2 * i + 1
            pos_index_next_car = 2 * (i + 1)
            vel_index_next_car = 2 * (i + 1) + 1
        elif order == 'position_first':
            pos_index_this_car = i
            vel_index_this_car = i + n
            pos_index_next_car = i + 1
            vel_index_next_car = i + 1 + n
        elif order == 'velocity_first':
            pos_index_this_car = i + n
            vel_index_this_car = i
            pos_index_next_car = i + 1 + n
            vel_index_next_car = i + 1

        # Position derivative w.r.t. velocity
        A[pos_index_this_car, vel_index_this_car] = 1.0

        # Velocity derivative w.r.t. position
        if i < n - 1:
            A[vel_index_this_car, pos_index_this_car] = -alpha_i / tau_i
            A[vel_index_this_car, pos_index_next_car] = alpha_i / tau_i

        # Velocity derivative w.r.t. velocity
        if i < n - 1:
            A[vel_index_this_car, vel_index_this_car] = -alpha_i - beta_i
            A[vel_index_this_car, vel_index_next_car] = beta_i

    return A

def stamp_dynamics_nonlinear(
    x: np.ndarray,
    p: list[Parameters], 
) -> np.ndarray:
    """
    Stamps the Jacobian of the nonlinear portion of the dynamics of the system.
    
    Inputs:
    x: numpy array of shape (2n,), current state vector.
    p: list of length (n,) Parameters objects, parameters vector.
    Outputs:
    Jf: numpy array of shape (2n, 2n), Jacobian of the nonlinear portion of the dynamics at state x.
    """
    n = len(p)
    Jf = np.zeros((2*n, 2*n))

    for i in range(n):
        alpha_i, beta_i, tau_i, K_i, L_i, d0_i = p[i].alpha, p[i].beta, p[i].tau, p[i].K, p[i].L, p[i].d0

        pos_index_this_car = 2 * i
        vel_index_this_car = 2 * i + 1
        pos_index_next_car = 2 * (i + 1)

        # Only compute if there is a car in front
        if i < n - 1:
            delta_x = x[pos_index_next_car] - x[pos_index_this_car]
            exp_term = np.exp(-L_i * (delta_x - d0_i))
            term_derivative = K_i * L_i * exp_term

            # Apply alpha/tau scaling as in eval_f
            val = (alpha_i / tau_i) * term_derivative

            # Update Jacobian
            # Derivative w.r.t z_i (this car position)
            Jf[vel_index_this_car, pos_index_this_car] = -val
            
            # Derivative w.r.t z_{i+1} (next car position)
            Jf[vel_index_this_car, pos_index_next_car] = val

    return Jf