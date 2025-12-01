import numpy as np

def gen_eval_u_cos(T):
    """
    Generates a function that evaluates the cosine input u at time t for the Periodic Steady State problem 2 in PS5
    
    Input:
    T   period of the cosine waveform
    
    Output:
    eval_u_cos_func   function that takes time t as input and returns u = cos((2*pi/T) * t)
    
    EXAMPLES:
    eval_u_cos_func = gen_eval_u_cos(T)
    u = eval_u_cos_func(t)
    """
    
    def eval_u_cos_func(t):
        u = np.cos((2 * np.pi / T) * t)
        return u
    
    return eval_u_cos_func
