import numpy as np
from tqdm import tqdm
from typing import Literal
from our_solvers.newtonNd import newtonNd

def trapezoidal_adaptive(eval_f, x_start, p, eval_u, t_start, t_stop, initial_timestep,
                         atol=1e-4, rtol=1e-4,  # <-- NEW: Tolerances for adaptation
                         errf=1e-8, errDeltax=1e-8, relDeltax=1e-8, MaxIter=30,
                         FiniteDifference=1, Jf_linear=None, Jf_eval_nonlinear=None, use_tqdm=False, 
                         min_step_size=0.05,
                         newton_linear_solver: Literal["LU", "solve_banded", "tgcr"] = "LU", Jf_bandwidth = None):
    """
    Fully implicit trapezoidal integrator using newtonNd. Adapted for our system
    to allow for a constant linear portion and a variable nonlinear portion of the Jacobian.
    Uses an adaptive time-stepping strategy based on local truncation error estimates.
    If Jf_eval is provided it should be the analytic Jacobian of eval_f:
        Jf_eval(x,p,u) -> df/dx
    In that case, we build the trapezoidal residual Jacobian:
        J_res = I - (h/2) * Jf_eval(x_next, p, u_{n+1})
    and pass FiniteDifference=0 to newtonNd, with Jfhand = J_res function wrapper.
    """
    X_list = [x_start.copy()]
    t_list = [t_start]
    
    t_curr = t_start
    h = initial_timestep
    x_curr = x_start.copy()

    # residual functions (Unchanged)
    def trap_residual(x_next, p_local, u_bundle):
        x_n, f_n, h_local, t_n1 = u_bundle
        u_n1 = eval_u(t_n1)
        f_n1 = eval_f(x_next, p_local, u_n1)
        return x_next - x_n - 0.5 * h_local * (f_n + f_n1)

    I_n = np.eye(len(x_start))
    def trap_J_res(x_next, p_local, u_bundle):
        _, _, h_local, t_n1 = u_bundle
        u_n1 = eval_u(t_n1)
        Jf = Jf_eval_nonlinear(x_next, p_local, u_n1) + Jf_linear
        return I_n - 0.5 * h_local * Jf

    pbar = tqdm(total=t_stop - t_start) if use_tqdm else None
    
    while t_curr < t_stop:
        # Don't step past t_stop
        if t_curr + h > t_stop:
            h = t_stop - t_curr

        u_n = eval_u(t_curr)
        f_n = eval_f(x_curr, p, u_n)

        # 1. Predictor (Forward Euler)
        x_pred = x_curr + h * f_n
        
        # Bundle data for residualeval_Jf_analytic_nonlinear
        t_next_candidate = t_curr + h
        u_bundle = (x_curr, f_n, h, t_next_candidate)

        # 2. Corrector (Newton Solve)
        x_next, converged, _, _, _, _, _ = newtonNd(
            fhand=trap_residual, 
            x0=x_pred, 
            p=p, 
            u=u_bundle,
            errf=errf, 
            errDeltax=errDeltax, 
            relDeltax=relDeltax, 
            MaxIter=MaxIter,
            FiniteDifference=FiniteDifference,
            Jfhand=trap_J_res if (Jf_eval_nonlinear is not None and Jf_linear is not None) else None,
            linearSolver=newton_linear_solver, 
            Jf_bandwidth=Jf_bandwidth
        )

        # 3. Error Estimation (LTE based on Predictor-Corrector difference)
        # We calculate a weighted norm of the difference between Euler and Trap
        scale = atol + rtol * np.maximum(np.abs(x_curr), np.abs(x_next))
        error_norm = np.linalg.norm((x_next - x_pred) / scale) / np.sqrt(len(x_curr))
        
        # 4. Controller Logic
        # Order 2 (Trapezoidal) -> exponent 1/2
        # Avoid division by zero with small epsilon
        adaptive_factor = (1.0 / (error_norm + 1e-10)) ** 0.5

        if (converged and error_norm <= 1.0) or h <= min_step_size:
            # === ACCEPT STEP ===
            t_curr = t_next_candidate
            x_curr = x_next
            
            X_list.append(x_curr)
            t_list.append(t_curr)
            
            if pbar: pbar.update(h)

            # Increase step size for next time (limit growth to 5x to be safe)
            if adaptive_factor > 1.0:
                h = h * min(adaptive_factor, 5.0)
                
        else:
            # === REJECT STEP ===
            # Newton failed OR error was too high. Decrease h and retry.
            # Limit decrease to 0.1x to prevent stalling
            if adaptive_factor < 1.0:
                h = h * max(adaptive_factor, 0.1)
            h = max(h, min_step_size)  # Ensure we don't go below min step size

    if pbar: pbar.close()

    return np.array(X_list).T, np.array(t_list)