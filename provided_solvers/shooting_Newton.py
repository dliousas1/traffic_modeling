import numpy as np
from .newtonNd import newtonNd
from .newtonNdGCR import newtonNdGCR
from .eval_f_Shooting import eval_f_Shooting

def shooting_Newton(eval_f, x0, p, eval_u, dt, errf, errDeltax, relDeltax, MaxIter, visualize, FiniteDifference, eval_Jf=None, use_GCR=False):
    """
    evaluates the value of the function that ShootingNewton needs to force to zero
    
    INPUTS:
    x0        : initial state guess for the periodic steady state       
    eval_f    : the function f(x, p, u) of the model dx/dt = f(x, p, u)
    eval_Jf   : [optional] if not given, Finite Difference will be used
    eval_u    : the function providing the periodic input to f(x, p, u)  
    dt        : timestep for the ODE integrator
    
    OUTPUTS:
    X_pss       : a matrix whose columns are states along the periodic steady state
    t_pss       : a vector with the corresponding times along the periodic steady state
    converged   : 1 if converged, 0 if not converged
    errf_k      : ||x(T) - x0||
    errDeltax_k : ||x0_k - x0_{k-1}||
    relDeltax_k : ||x0_k - x0_{k-1}|| / ||x0_k||
    iterations  : number of Newton iterations k to get to convergence
    """

    # Setup parameters for ShootingNewton
    p['eval_f'] = eval_f
    if not FiniteDifference and eval_Jf is not None:
        p['eval_Jf'] = eval_Jf  # This is just to help the ODE integrator, not ShootingNewton
    p['eval_u'] = eval_u
    p['dt'] = dt

    # Note: we don't have an eval_JF_Shooting function, so Finite Difference is always used for ShootingNewton
    FiniteDifferenceShooting = 1

    if not use_GCR:
        # Solve using NewtonNd
        x0_pss, converged, errf_k, errDeltax_k, relDeltax_k, iterations, _ = newtonNd(
            eval_f_Shooting, x0, p, np.nan, errf, errDeltax, relDeltax, MaxIter, visualize, FiniteDifferenceShooting, None
        )
    else:
        # Solve using NewtonNdGCR
        x0_pss, converged, errf_k, errDeltax_k, relDeltax_k, iterations, _ = newtonNdGCR(
            eval_f_Shooting, x0, p, np.nan, errf, errDeltax, relDeltax, MaxIter, visualize, FiniteDifferenceShooting, None,
            tolrGCR=1e-8, epsMF=1e-8
        )


    if converged:
        # If Newton method converged, evaluate the periodic steady state
        F, X_pss, t_pss = eval_f_Shooting(x0_pss, p)
    else:
        print('ShootingNewton reached the maximum allowed number of iterations without converging')
        print(f"errf_k: {errf_k}, errDeltax_k: {errDeltax_k}, relDeltax_k: {relDeltax_k}, iterations: {iterations}")
        X_pss, t_pss = None, None

    return X_pss, t_pss, converged, errf_k, errDeltax_k, relDeltax_k, iterations
