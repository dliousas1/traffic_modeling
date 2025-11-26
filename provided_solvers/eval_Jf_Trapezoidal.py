import numpy as np

def eval_Jf_Trapezoidal(x_next, p, u_next):
    """
    evaluates the Jacobian matrix required by Trapezoidal
    i.e. JF_Trapezoidal = 
    the name of the file containing function f( ) is passed in p['eval_Jf']
    
    JF_Trapezoidal = eval_JF_Trapezoidal(x_next, p, u_next)
    """
    
    N = len(x_next)
    JF_Trapezoidal = np.eye(N) - 0.5 * p['dt'] * p['eval_Jf'](x_next, p, u_next)
    return JF_Trapezoidal
