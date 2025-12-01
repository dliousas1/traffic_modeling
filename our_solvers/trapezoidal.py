import numpy as np
from tqdm import tqdm
from typing import Literal

from our_solvers.newtonNd import newtonNd


def trapezoidal(eval_f, x_start, p, eval_u, t_start, t_stop, timestep,
                errf=1e-8, errDeltax=1e-8, relDeltax=1e-8, MaxIter=30,
                FiniteDifference=1, Jf_linear=None, Jf_eval_nonlinear=None, use_tqdm=False, 
                newton_linear_solver: Literal["LU", "solve_banded", "tgcr"] = "LU", Jf_bandwidth = None):
    """
    Fully implicit trapezoidal integrator using newtonNd. Adapted for our system
    to allow for a constant linear portion and a variable nonlinear portion of the Jacobian.
    If Jf_eval is provided it should be the analytic Jacobian of eval_f:
        Jf_eval(x,p,u) -> df/dx
    In that case, we build the trapezoidal residual Jacobian:
        J_res = I - (h/2) * Jf_eval(x_next, p, u_{n+1})
    and pass FiniteDifference=0 to newtonNd, with Jfhand = J_res function wrapper.
    """

    num_steps = int(np.ceil((t_stop - t_start) / timestep)) + 1
    X = np.zeros((len(x_start), num_steps))
    t = np.zeros(num_steps)

    X[:, 0] = x_start.copy()
    t[0] = t_start
    h = timestep

    # residual: f_res(x_next, p, u_bundle)
    # u_bundle = (x_n, f_n, h, t_n1)
    def trap_residual(x_next, p_local, u_bundle):
        x_n, f_n, h_local, t_n1 = u_bundle
        u_n1 = eval_u(t_n1)
        f_n1 = eval_f(x_next, p_local, u_n1)
        return x_next - x_n - 0.5 * h_local * (f_n + f_n1)

    # If analytic Jacobian available, provide Jacobian of residual:
    # J_res(x_next) = I - (h/2) * Jf(x_next)
    I_n = np.eye(len(x_start))
    def trap_J_res(x_next, p_local, u_bundle):
        _, _, h_local, t_n1 = u_bundle
        u_n1 = eval_u(t_n1)
        Jf = Jf_eval_nonlinear(x_next, p_local, u_n1) + Jf_linear
        return I_n - 0.5 * h_local * Jf

    # time-stepping loop
    if use_tqdm:
        iterator = tqdm(range(num_steps - 1))
    else:
        iterator = range(num_steps - 1)

    for n in iterator:
        t_n = t[n]
        t_n1 = t_n + h
        t[n+1] = t_n1

        x_n = X[:, n]
        u_n = eval_u(t_n)
        f_n = eval_f(x_n, p, u_n)

        # initial guess: forward euler
        x0 = x_n + h * f_n

        # bundle data for residual
        u_bundle = (x_n, f_n, h, t_n1)

        # Solve the nonlinear system
        x_next, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X_hist = newtonNd(
            fhand=trap_residual,
            x0=x0,
            p=p,
            u=u_bundle,
            errf=errf,
            errDeltax=errDeltax,
            relDeltax=relDeltax,
            MaxIter=MaxIter,
            FiniteDifference=FiniteDifference,
            Jfhand=trap_J_res if (Jf_eval_nonlinear is not None and Jf_linear is not None) else None ,
            linearSolver=newton_linear_solver,
            Jf_bandwidth=Jf_bandwidth
        )

        if not converged:
            print(f"Warning: Newton did not converge at step {n} (t={t_n} -> t={t_n1}). "
                  f"errf={errf_k:.2e}, errDeltax={errDeltax_k:.2e}, relDeltax={relDeltax_k:.2e}")

        X[:, n+1] = x_next

    return X, t