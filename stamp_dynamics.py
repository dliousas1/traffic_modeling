import numpy as np
from evaluate_f import Parameters

def stamp_dynamics(p):
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

        # Position derivative w.r.t. velocity
        A[2*i, 2*i + 1] = 1.0

        # Velocity derivative w.r.t. position
        if i < n - 1:
            A[2*i + 1, 2*i] = -alpha_i / tau_i
            A[2*i + 1, 2*(i + 1)] = alpha_i / tau_i
        else:
            # For the first car (last in the list), no reaction to anyone ahead
            A[2*i + 1, 2*i] = 0.0

        # Velocity derivative w.r.t. velocity
        if i < n - 1:
            A[2*i + 1, 2*i + 1] = -alpha_i - beta_i
            A[2*i + 1, 2*(i + 1) + 1] = beta_i
        else:
            # For the first car, velocity derivative is 0
            A[2*i + 1, 2*i + 1] = 0.0

    return A
