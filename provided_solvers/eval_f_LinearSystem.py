import numpy as np
from scipy.sparse import issparse

def eval_f_LinearSystem(x, p, u):
    """
    Evaluates the vector field f(x, p, u) 
    at state vector x, and with vector of inputs u.
    p is a structure containing all model parameters
    i.e. in this case: matrices p.A and p.B 
    corresponding to state space model dx/dt = p.A x + p.B u

    f = eval_f_LinearSystem(x, p, u)
    """

    # Ensure x and u are column vectors
    x = np.reshape(x, (-1, 1)) if len(x.shape) == 1 else x
    u = np.reshape(u, (-1, 1)) if len(np.array(u).shape) == 1 else u
    f = p['A'].dot(x) + np.reshape(p['B'].dot(u), (-1, 1))
    if issparse(f):
        f = f.toarray()
    else:
        f = np.asarray(f)
    
    return f.squeeze()
