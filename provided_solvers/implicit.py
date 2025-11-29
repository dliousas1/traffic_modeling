import numpy as np
from .newtonNd import newtonNd
from .eval_f_BackwardEuler import eval_f_BackwardEuler
from .eval_f_Trapezoidal import eval_f_Trapezoidal
from .eval_Jf_BackwardEuler import eval_Jf_BackwardEuler
from .eval_Jf_Trapezoidal import eval_Jf_Trapezoidal
from .visualize_state import visualize_state
from .newtonNdGCR import newtonNdGCR
import matplotlib.pyplot as plt

def implicit(method, eval_f, x_start, p, eval_u, t_start, t_stop, timestep, visualize, FiniteDifference, eval_Jf=None, use_GCR=False):
    """
    Uses an Implicit ODE integration method 
    to simulate the state-space model dx/dt = f(x, p, u).
    to simulate states model dx/dt=f(x,p,u)
    from state x_start at time t_start
    until time t_stop, with time intervals timestep
    eval_f is a string containing the name of the function that evaluates f(x,p,u)
    eval_u is a string containing the name of the funciton that evaluates u(t)
    It uses separatly defined functions val_f_BackwardEuler, eval_f_trapezoidal
    [and corresponding Jacobians eval_Jf_BackwardEuler, eval_Jf_trapezoidal]
    to define the implicit method (e.g. Backward Euler or Trapezoidal etc.)
    possible values for method are: 'BackwardEuler' and 'Trapezoidal'

    EXAMPLE:
    X, t, k = implicit('BackwardEuler', eval_f, x_start, p, eval_u, t_start, t_stop, timestep, visualize, FiniteDifference, eval_Jf)

    """

    # this is for a quick less accurate solution
    #errf_implicit = 1e-3
    #errDeltax     = 1e-3
    #relDeltax     = 0.01
    #MaxIter       = 10

    # use this in most cases
    errf_implicit = 1e-8
    errDeltax = 1e-8
    relDeltax = 1e-8
    MaxIter = 6  # if not converged in MaxIter, dt will be reduced automatically

    # use this for PS5p1
    #errf_implicit = 1e-10  # the drift diffusion units are a bit smaller than usual
    #errDeltax = 1e-8
    #relDeltax = 1  # note this is equivalent to NOT specifying it
    #MaxIter = 6  # the exponentials in the drift diffusion are very nonlinear

    # Assign functions based on the method
    if method == 'BackwardEuler':
        eval_f_implicit = eval_f_BackwardEuler
        eval_Jf_implicit = eval_Jf_BackwardEuler
    elif method == 'Trapezoidal':
        eval_f_implicit = eval_f_Trapezoidal
        eval_Jf_implicit = eval_Jf_Trapezoidal
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'BackwardEuler' or 'Trapezoidal'.")

    # Update parameters with provided functions
    p['eval_f'] = eval_f
    if not FiniteDifference and eval_Jf is not None:
        p['eval_Jf'] = eval_Jf

    X = [x_start]
    t = [t_start]
    k = []
    n = 1

    # Initialize visualization
    if visualize:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        fig.show()

    while n <= np.ceil((t_stop - t_start) / timestep):
        p['x_prev'] = X[-1]
        p['u_prev'], _ = eval_u(t[-1]) if isinstance(eval_u(t[-1]), tuple) else (eval_u(t[-1]), None)
        p['dt'] = min(timestep, t_stop - t[-1])
        t_next = t[-1] + p['dt']
        u_next, _ = eval_u(t_next) if isinstance(eval_u(t_next), tuple) else (eval_u(t_next), None)

        # Initial guess for Newton method
        x0 = X[-1]  # this is the easiest guess but could also use one step of Forward Euler

        if use_GCR:
            # Solve with Newton-GCR method
            X_next, converged, errF_k, errDeltax_k, relDeltax_k, k_iter, _ = newtonNdGCR(
                eval_f_implicit, x0, p, u_next, errf_implicit, errDeltax, relDeltax, MaxIter, visualize=False,
                FiniteDifference=False,  jfhand=None, tolrGCR=1e-8, epsMF=1e-8
            )
        else:
            # Solve with Newton method
            X_next, converged, errF_k, errDeltax_k, relDeltax_k, k_iter, _ = newtonNd(
                eval_f_implicit, x0, p, u_next, errf_implicit, errDeltax, relDeltax, MaxIter, visualize=False,
                FiniteDifference=FiniteDifference, Jfhand=eval_Jf_implicit
            )


        

        if not converged:
            timestep /= 2  # Halve the timestep if Newton did not converge
            print(f"At t={t_next:.4f}, Newton did not converge: decreasing stepsize to {timestep:.4e}")
            continue
        else:
            X.append(X_next)
            t.append(t_next)
            k.append(k_iter)

            # Update plot
            if visualize:
                ax_top, ax_bottom = visualize_state(np.array(t), np.array(X).T, n, 'r.', ax_top, ax_bottom)
                plt.pause(0.001)

        n += 1

    if visualize:
        plt.show()

    return np.array(X).T, np.array(t), np.array(k)
