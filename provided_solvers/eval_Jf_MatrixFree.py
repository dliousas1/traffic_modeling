import math
import numpy as np

def eval_Jf_MatrixFree(eval_f,x0,p,u,dx,epsMF=None):
    """
    Evaluates the Matrix-Free product between the jacobian of the vector field f() at state x0 and the vector dx
    INPUTS
    eval_f  : Name of the function that evalautes f()
    x0      : Current state for the jacobian: Jf(x0)
    p       : Structure containing all the model parameters for f()
    u       : Vector containining the values of the system inputs
    dx      : Vector to be multiplied by Jacobian 
    epsMF   : (optional) Finite Difference perturbation for matrix free directional derivative. if not Specified, uses (2 * sqrt(eps) / ||dx||) * max(1,||x0||), where eps is the relative machine precision
    
    Inspired by taylor series expansion, computes a directional derivative as a finite difference of function evaluations perturbed by scalar epsMF
    Jfdx = (f(x0+epsMF*dx) - f(x0)) / epsMF
    """
    if epsMF is None:
        eps = np.finfo(float).eps
        #epsMF =   sqrt(eps);   # terrible when ||x0|| or ||dx|| large
        #epsMF =   sqrt(eps)             /norm(dx,inf);      # just ok
        #epsMF =   sqrt(eps)*(1+norm(x0,inf))/norm(dx,inf); # great
        epsMF = 2*math.sqrt(eps)*(1+np.linalg.norm(x0,np.inf))/np.linalg.norm(dx,np.inf); # great  USE THIS
        #epsMF = 2*sqrt(eps*(1+norm(x0,inf)/norm(dx,inf))); # not good
        #epsMF = 2*sqrt(eps*(1+norm(x0,inf)))/norm(dx,inf); # NITSOL almost good
        # same as 2*sqrt(eps)*max(1,||x0||)/||dx||;
        print("epsMF not specified using: 2*sqrt(eps)*(1+||x0||)/||dx|| = ", epsMF)
        
    f0 = eval_f(x0,p,u)
    fepsMF = eval_f(x0+epsMF*dx,p,u)
    Jfdx = (fepsMF - f0)/epsMF
    return Jfdx,epsMF