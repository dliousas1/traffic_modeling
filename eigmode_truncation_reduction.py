import numpy as np
import os
import time
import matplotlib.pyplot as plt
from icecream import ic

# Import your local modules
from evaluate_f import Parameters
from evaluate_Jf import eval_Jf_analytic_linear
from our_solvers.trapezoidal_adaptive import trapezoidal_adaptive

def eigenmode_truncation(A, b, C_full, q):
    vals, V = np.linalg.eig(A)
    
    # Standard: Keep Slow modes
    metric = np.abs(vals)
        
    # --- 2. Sort and Select ---
    # Sort DESCENDING (Keep largest metric)
    idx = np.argsort(metric)
    
    basis_vectors = []
    current_rank = 0
    processed_conjugates = set() 
    
    # Deterministic Construction (Prevents splitting complex pairs)
    for i in idx:
        if current_rank >= q:
            break
        val = vals[i]
        vec = V[:, i]
        
        if i in processed_conjugates: continue

        if np.iscomplex(val):
            if current_rank + 2 > q: break 
            basis_vectors.append(vec.real)
            basis_vectors.append(vec.imag)
            current_rank += 2
            
            conj_candidates = np.where(np.isclose(vals, np.conj(val)))[0]
            for c in conj_candidates: 
                if c != i: processed_conjugates.add(c)
        else:
            if current_rank + 1 > q: break
            basis_vectors.append(vec.real)
            current_rank += 1

    basis_pool = np.column_stack(basis_vectors)
    Q_basis, _ = np.linalg.qr(basis_pool)
    T = Q_basis[:, :current_rank] 
    
    # Project Matrices
    Ar = T.T @ A @ T
    br = T.T @ b
    Cr = C_full @ T
    
    return T, Ar, br, Cr

def compute_total_accel(velocities, t):
    accelerations = np.zeros_like(velocities)
    for i in range(1, velocities.shape[1]):
        dt_sim = t[i] - t[i - 1]
        accelerations[:, i] = (velocities[:, i] - velocities[:, i - 1]) / dt_sim

    accelerations = np.sum(np.abs(accelerations), axis=0)
    accel_integral = np.trapz(accelerations, t)

    return accelerations, accel_integral

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    np.random.seed(42)

    # --- Setup ---
    n_cars = 100
    n_states = 2 * n_cars
    
    parameters = []
    for i in range(n_cars):
        parameters.append(Parameters(
            alpha=np.random.normal(1.0, 0.1), beta=np.random.normal(1.0, 0.1),
            tau=np.random.normal(1.0, 0.2), K=np.random.normal(1.0, 0.5),
            L=np.random.normal(0.5, 0.1), d0=np.random.normal(2.0, 0.5)
        ))
    param_dict = {"parameters": parameters}

    # Initial state: Position ramp + Random velocity (Creates the shockwave)
    x0 = np.zeros(2 * n_cars)
    for i in range(n_cars):
        x0[2*i] = (i * 15.0) 
        x0[2*i + 1] = np.random.normal(3.0, 1.0) 
    
    print("Generating Full System Matrices...")
    A = eval_Jf_analytic_linear(x0, param_dict, n_cars)
    B = np.zeros((n_states, 1)); B[-1, 0] = 3.0 
    C_full = A[1::2, :] 

    # --- Run Comparisons ---
    # We compare at q=50 (Medium reduction) and q=20 (Aggressive)
    q_targets = [190, 150, 100]
    
    # 1. Run FULL Model once as Ground Truth
    print("\nSimulating FULL Model (Ground Truth)...")
    start = time.time()
    x_full, t_full = trapezoidal_adaptive(
        eval_f=lambda x, _, __: A @ x, x_start=x0, p=param_dict, eval_u=lambda t: None,
        t_start=0, t_stop=30.0, initial_timestep=0.05,
        errf=1e-6, errDeltax=1e-6, relDeltax=1e-6, MaxIter=50, FiniteDifference=0,
        Jf_linear=A, Jf_eval_nonlinear=None, use_tqdm=True, 
        newton_linear_solver="solve_banded", Jf_bandwidth=(1, 2)
    )
    y_full, integral_full = compute_total_accel(x_full[1::2, :], t_full)
    print(f"Full simulation took {time.time() - start:.2f}s")

    # 2. Run Reduced Models
    results = {}
    
    for q in q_targets:
        results[q] = {}
        print(f"\n--- Reducing to q={q} ---")
        
        # A) Eigenmode Truncation (using optimal H2 metric)
        T_eig, A_eig, _, C_eig = eigenmode_truncation(A, B, C_full, q)
        x0_eig = T_eig.T @ x0
        
        # Simulate Eigen
        # print(f"  Simulating Eigen-Reduced...")
        x_e, t_e = trapezoidal_adaptive(
            eval_f=lambda x, _, __: A_eig @ x, x_start=x0_eig, p=param_dict, eval_u=lambda t: None,
            t_start=0, t_stop=30.0, initial_timestep=0.05,
            errf=1e-6, errDeltax=1e-6, relDeltax=1e-6, MaxIter=50, FiniteDifference=0,
            Jf_linear=A_eig, Jf_eval_nonlinear=None, use_tqdm=False, newton_linear_solver="solve"
        )
        x_e = T_eig @ x_e  # Lift back to full space for comparison
        results[q] = (t_e, *compute_total_accel(x_e[1::2, :], t_e))

    # --- Plotting ---
    print("\nGenerating Comparison Plot...")
    fig, axes = plt.subplots(len(q_targets), 1, figsize=(10, 5 * len(q_targets)), sharex=True)
    if len(q_targets) == 1: axes = [axes]

    for i, q in enumerate(q_targets):
        ax = axes[i]
        # Plot Full
        ax.plot(t_full, y_full, 'k-', linewidth=3, alpha=0.3, label='Full Model (N=200)')
        
        # Plot Reduced
        t_e, y_e, integral_e = results[q]
        ax.plot(t_e, y_e, 'r--', linewidth=1.5, label=f'Eigen Truncation (q={q})')
        
        ax.set_ylabel('Total Accel (m/s²)')
        ax.set_title(f'Comparison at Reduced Order q={q}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"q={q}: Integral Full={integral_full:.2f}, Integral Eigen={integral_e:.2f}")

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()