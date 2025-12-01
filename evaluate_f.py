import numpy as np


class Parameters:
    def __init__(self, alpha, beta, tau, K, L, d0, input_ampl=0.0):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.K = K
        self.L = L
        self.d0 = d0
        self.input_ampl = input_ampl

def eval_f(x, param_dict, u=0.0):
    """
    Computes the dynamics function f(x, p) for a given state vector 
    x and parameters vector p.

    Inputs:
    x: numpy array of length (2n,).
    p: list of length (n,) Parameters objects, parameters vector.

    Outputs:
    f: numpy array of shape (2n,), dynamics function evaluated at x and p.
    """

    p = param_dict['parameters']
    n = len(p)
    assert len(x) == 2 * n, "State vector x must have length 2 * number of cars (n)."

    x = np.array(x)
    
    # Preallocate f
    f = np.zeros(2 * n)
    for i in range(n):
        z_i, v_i = x[2*i], x[2*i + 1]
        f[2*i] = v_i.item()

        # If this is the last car, it has no car in front of it
        if i == n - 1:
            a_i = np.array([0.0])

        # Else, follow the car in front
        else:
            j = i + 1
            z_j, v_j = x[2*j], x[2*j + 1]
            alpha_i, beta_i, tau_i, K_i, L_i, d0_i = p[i].alpha, p[i].beta, p[i].tau, p[i].K, p[i].L, p[i].d0

            a_i = (alpha_i/tau_i) * (z_j - z_i - K_i * np.exp(-L_i * (z_j - z_i - d0_i))) + beta_i * (v_j - v_i) - alpha_i * v_i
        
        f[2*i + 1] = a_i.item()
        if u is not None:
            f[2*i + 1] += p[i].input_ampl * u

    return f