class State:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

class Parameters:
    def __init__(self, alpha, beta, tau):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

def eval_f(x, p):
    """
    Computes the dynamics function f(x, p) for a given state vector 
    x and parameters vector p.

    Inputs:
    x: list of length (n,) State objects, state vector.
    p: list of length (n,) Parameters objects, parameters vector.

    Outputs:
    f: numpy array of shape (2n,), dynamics function evaluated at x and p.
    """
    f = []
    for i in range(len(x)):
        z_i, v_i = x[i].position, x[i].velocity
        
        # If this is the last car, it has no car in front of it
        if i == len(x) - 1:
            a_i = 0.0

        # Else, follow the car in front
        else:
            j = i + 1

            z_j, v_j = x[j].position, x[j].velocity
            alpha_i, beta_i, tau_i = p[i].alpha, p[i].beta, p[i].tau

            a_i = (alpha_i/tau_i) * (z_j - z_i) + beta_i * (v_j - v_i) - alpha_i * v_i

        f.extend([v_i, a_i])
    
    return f