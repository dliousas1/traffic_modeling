import numpy as np

def eval_Jf_BackwardEuler(x_next, p, u_next):
    """
    evaluates the Jacobian matrix required by Backward Euler
    i.e. JF_BackwardEuler = Identity - p['dt'] * p['eval_Jf'](x_next, p, u_next)
    the name of the file containing function f( ) is passed in p['eval_Jf']
    
    JF_BackwardEuler = eval_Jf_BackwardEuler(x_next, p, u_next)
    """

    N = len(x_next)
    JF_BackwardEuler = np.eye(N) - p['dt'] * p['eval_Jf'](x_next, p, u_next)
    return JF_BackwardEuler
