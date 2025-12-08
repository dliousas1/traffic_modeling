import numpy as np
import math
from tqdm import tqdm
from typing import Literal
from provided_solvers.newtonNd import newtonNd

def trapezoidal_adaptive(eval_f, x_start, p, eval_u, t_start, t_stop, initial_timestep,
                         errf=1e-8, errDeltax=1e-8, relDeltax=1e-8, MaxIter=30,
                         FiniteDifference=1, Jf_eval=None, use_tqdm=False, 
                         min_step_size=0.01,
                         return_full_traj=False,
                         newton_linear_solver: Literal["LU", "solve_banded", "tgcr"] = "LU", Jf_bandwidth = None):
    
    # --- 1. PRE-ALLOCATION ---
    max_steps = int(math.ceil((t_stop - t_start) / min_step_size)) + 10
    n_states = len(x_start)
    
    # Initialize arrays with zeros
    # Shape is (n_states, max_steps) to match the desired output format .T
    X_array = np.zeros((n_states, max_steps))
    t_array = np.zeros(max_steps)
    
    # --- 2. INITIALIZATION ---
    # Store the initial state at index 0
    store_idx = 0
    X_array[:, store_idx] = x_start
    t_array[store_idx] = t_start
    store_idx += 1
    
    # Track the last time we logged data (for sparse logging)
    last_log_time = t_start

    t_curr = t_start
    h = initial_timestep
    x_curr = x_start.copy()
    
    # Adams-Bashforth setup
    u_start = eval_u(t_start)
    f_curr = eval_f(x_curr, p, u_start)
    f_prev = None 
    h_prev = initial_timestep 
    
    I_n = np.eye(n_states)
    
    # --- Residual Definitions ---
    def trap_residual(x_next, p_local, u_bundle):
        x_n, f_n, h_local, t_n1 = u_bundle
        u_n1 = eval_u(t_n1)
        f_n1 = eval_f(x_next, p_local, u_n1)
        return x_next - x_n - 0.5 * h_local * (f_n + f_n1)

    def trap_J_res(x_next, p_local, u_bundle):
        _, _, h_local, t_n1 = u_bundle
        u_n1 = eval_u(t_n1)
        Jf = Jf_eval(x_next, p_local, u_n1) if Jf_eval is not None else None
        return I_n - 0.5 * h_local * Jf

    pbar = tqdm(total=t_stop - t_start) if use_tqdm else None
    step_was_rejected = False

    # --- 3. MAIN LOOP ---
    while t_curr < t_stop:
        # Cap h to hit t_stop exactly
        if t_curr + h > t_stop:
            h = t_stop - t_curr
        else:
            h = max(h, min_step_size)

        # Predictor (Adams-Bashforth 2)
        if f_prev is not None:
            ratio = h / (2 * h_prev)
            x_pred = x_curr + h * ((1 + ratio) * f_curr - ratio * f_prev)
        else:
            x_pred = x_curr + h * f_curr

        # Corrector (Newton)
        t_next_candidate = t_curr + h
        u_bundle = (x_curr, f_curr, h, t_next_candidate)

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
            Jfhand=trap_J_res,
            linearSolver=newton_linear_solver, 
            Jf_bandwidth=Jf_bandwidth
        )

        # Error Estimation
        scale = errDeltax + relDeltax * np.maximum(np.abs(x_curr), np.abs(x_next))
        error_norm = np.linalg.norm((x_next - x_pred) / scale) / np.sqrt(n_states)
        
        # Controller Logic
        safety_factor = 0.9 # Slightly safer than 1.0 to reduce rejection loops
        
        if (converged and error_norm <= 1.0) or h <= min_step_size:
            # === ACCEPT STEP ===
            t_curr = t_next_candidate
            x_curr = x_next
            
            f_prev = f_curr
            u_next = eval_u(t_curr)
            f_curr = eval_f(x_curr, p, u_next)
            h_prev = h
            
            if pbar: pbar.update(h)

            # --- 4. ARRAY FILLING LOGIC ---
            # Check logging condition
            should_log = False
            if return_full_traj:
                should_log = True
            elif t_curr - last_log_time >= 0.1: # Sparse logging (0.1s interval)
                should_log = True
            # Always force log if it's the very last step
            elif t_curr >= t_stop: 
                should_log = True
            
            if should_log:
                X_array[:, store_idx] = x_curr
                t_array[store_idx] = t_curr
                last_log_time = t_curr
                store_idx += 1

            # Step Size Adjustment
            scale_factor = safety_factor * (1.0 / (error_norm + 1e-10)) ** (1.0/2.0)
            scale_factor = min(scale_factor, 2.0)
            scale_factor = max(scale_factor, 0.5)

            if step_was_rejected:
                # If we just recovered, keep h steady to stabilize
                h = h 
                step_was_rejected = False
            else:
                h = h * scale_factor
        else:
            # === REJECT STEP ===
            step_was_rejected = True
            
            if not converged:
                scale_factor = 0.5
            else:
                scale_factor = safety_factor * (1.0 / (error_norm + 1e-10)) ** (1.0/2.0)
            
            scale_factor = max(scale_factor, 0.1) 
            h = h * scale_factor

    if pbar: pbar.close()

    # --- 5. TRIM AND RETURN ---
    # Slice the arrays to return only the filled portion
    return X_array[:, :store_idx], t_array[:store_idx]