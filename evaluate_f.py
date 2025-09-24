import numpy as np


class Parameters:
    def __init__(self, alpha, beta, tau):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

def eval_f(x, p, u=None):
    """
    Computes the dynamics function f(x, p) for a given state vector 
    x and parameters vector p.

    Inputs:
    x: numpy array of length (2n,).
    p: list of length (n,) Parameters objects, parameters vector.

    Outputs:
    f: numpy array of shape (2n,), dynamics function evaluated at x and p.
    """
    n = len(p)
    assert len(x) == 2 * n, "State vector x must have length 2 * number of cars (n)."

    # Preallocate f
    f = np.zeros(2 * n)
    for i in range(n):
        z_i, v_i = x[2*i], x[2*i + 1]
        
        # If this is the last car, it has no car in front of it
        if i == n - 1:
            a_i = 0.0

        # Else, follow the car in front
        else:
            j = i + 1
            z_j, v_j = x[2*j], x[2*j + 1]
            alpha_i, beta_i, tau_i = p[i].alpha, p[i].beta, p[i].tau

            a_i = (alpha_i/tau_i) * (z_j - z_i) + beta_i * (v_j - v_i) - alpha_i * v_i

        f[2*i] = v_i
        f[2*i + 1] = a_i

    return f