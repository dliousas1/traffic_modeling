import numpy as np

from typing import Literal

from evaluate_f import Parameters


import numpy as np

from typing import Literal

from evaluate_f import Parameters


def stamp_dynamics(p: list[Parameters], order: Literal['alternate', 'position_first', 'velocity_first']='alternate') -> np.ndarray:
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

    # Drop first position (z1) and last car's position/velocity (zN, vN).
    # This protects against singularities by removing zero-pivots.
    if order == 'alternate':
        drop_rows_cols = {0, 2*(n-1), 2*(n-1)+1}
    elif order == 'position_first':
        drop_rows_cols = {0, n-1, 2*n-1}
    elif order == 'velocity_first':
        drop_rows_cols = {n, 2*n-1, n-1}

    keep = np.ones(2*n, dtype=bool)
    keep[list(drop_rows_cols)] = False
    A = A[np.ix_(keep, keep)]

    return A
