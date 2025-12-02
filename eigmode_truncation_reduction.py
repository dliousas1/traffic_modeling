import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.linalg import schur, eigvals
import tracemalloc

from evaluate_f import Parameters, eval_f
from evaluate_Jf import eval_Jf_analytic_linear
from our_solvers.trapezoidal import trapezoidal
from provided_solvers.SimpleSolver import SimpleSolver


def eigenmode_truncation(A, B, C_full, q, x0=None):
    """
    Robust Eigenmode Truncation using Schur Decomposition with Leader Pinning.
    """
    n = A.shape[0]

    # 1. Determine Cutoff (Sort Smallest -> Largest)
    mags = np.sort(np.abs(eigvals(A)))
    k = min(q, n)
    cutoff = mags[k - 1] + 1e-9

    # 2. Schur Decomposition
    _, Z, _ = schur(A, output='real', sort=lambda r, i: r**2 + i**2 <= cutoff**2)

    # 3. Truncate
    basis = Z[:, :k]

    # 4. Augmentation Step
    augmented_vectors = []
    
    # A) Leader Pinning (Crucial for traffic!)
    # Force the basis to contain the exact states of the lead car (Pos and Vel).
    # This prevents 'back-coupling' where follower drag leaks into the leader.
    # Lead Car Pos: Index 0, Lead Car Vel: Index 1
    leader_pos = np.zeros((n, 1)); leader_pos[-2] = 1.0
    leader_vel = np.zeros((n, 1)); leader_vel[-1] = 1.0
    augmented_vectors.extend([leader_pos, leader_vel])

    # B) Initial Condition Augmentation
    if x0 is not None:
        augmented_vectors.append(x0.reshape(-1, 1))
    
    # Apply Gram-Schmidt to add these vectors to the basis
    for vec in augmented_vectors:
        # Project vec onto current basis
        projection = basis @ (basis.T @ vec)
        residual = vec - projection
        
        # If vector implies a new direction, add it
        norm = np.linalg.norm(residual)
        if norm > 1e-9:
            basis = np.column_stack([basis, residual / norm])

    # 5. Final Orthogonalization & Project
    T, _ = np.linalg.qr(basis)
    return T, T.T @ A @ T, T.T @ B, C_full @ T

def compute_total_accel(velocities, t):
    accelerations = np.zeros_like(velocities)
    for i in range(1, velocities.shape[1]):
        dt_sim = t[i] - t[i - 1]
        accelerations[:, i] = (velocities[:, i] - velocities[:, i - 1]) / dt_sim

    accelerations = np.sum(np.abs(accelerations), axis=0)
    accel_integral = np.trapezoid(accelerations, t)

    return accelerations, accel_integral

def perform_eigmode_truncation_reduction(seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    np.random.seed(seed)

    # --- Setup ---
    n_cars = 100
    n_states = 2 * n_cars
    
    parameters = []
    for i in range(n_cars):
        parameters.append(Parameters(
            alpha=np.random.normal(1.0, 0.1), beta=np.random.normal(1.0, 0.1),
            tau=np.random.normal(1.0, 0.2), 
            K=0.0, L=0.0, d0=0.0
        ))
    
    parameters[-1].input_ampl = 5.0
    param_dict = {"parameters": parameters}

    # Initial state: Position ramp + Random velocity (Creates the shockwave)
    x0 = np.zeros(2 * n_cars)
    for i in range(n_cars):
        x0[2*i] = 15*i
        x0[2*i + 1] = 20.0 
    
    print("Generating Full System Matrices...")
    tracemalloc.start()
    start = time.time()
    A = eval_Jf_analytic_linear(x0, param_dict, n_cars)
    B = np.zeros((n_states,))
    B[-1] = 5.0
    C_full = A[1::2, :]
    end = time.time()
    print(f"Full system matrices generated in {end - start:.2f}s")
    current, peak_generation = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Full system matrices generated. Peak memory during generation: {peak_generation / (1024 * 1024):.2f} MB")

    eval_u = lambda t: 0

    # Simulation parameters
    t_start = 0.0
    t_stop = 30.0
    golden_reference_timestep = 1e-4

    # Generate the golden reference
    base_dir = os.path.dirname(__file__)
    golden_path = os.path.join(base_dir, "references/golden_reference_eigmode_truncation.npz")
    if not os.path.exists(golden_path):
        print("Computing golden reference solution...")
        
        # Simulate golden reference
        X_golden, t_golden = SimpleSolver(
            eval_f,
            x0,
            param_dict,
            eval_u=eval_u,
            NumIter=int((t_stop - t_start) / golden_reference_timestep),
            w=golden_reference_timestep,
            use_tqdm=True,
            visualize=False
        )

        # Simulate trajectory with slightly larger timestep to quantify confidence
        X_check, t_check = SimpleSolver(
            eval_f,
            x0,
            param_dict,
            eval_u=eval_u,
            NumIter=int((t_stop - t_start) / (golden_reference_timestep * 2)),
            w=golden_reference_timestep * 2,
            use_tqdm=True,
            visualize=False
        )

        # Compute the error between the two solutions at final time
        confidence_golden = np.linalg.norm(X_golden[:, -1] - X_check[:, -1], np.inf)
        print(f"Final error between golden reference and check solution: {confidence_golden:.6e}")
        
        # Save golden reference to disk for reuse
        os.makedirs(os.path.dirname(golden_path), exist_ok=True)
        np.savez_compressed(    
            golden_path,
            X_golden=X_golden,
            t_golden=t_golden,
            num_cars=n_cars,
            t_start=t_start,
            t_stop=t_stop,
            golden_timestep=golden_reference_timestep,
            x0=x0,
            confidence=confidence_golden,
        )
        print(f"Golden reference saved to: {golden_path}")
    else:
        data = np.load(golden_path)
        X_golden = data['X_golden']
        t_golden = data['t_golden']
        confidence_golden = data['confidence']
        print(f"Golden reference loaded from: {golden_path}")

    # Crop X_golden to be t_stop long
    end_idx = np.searchsorted(t_golden, t_stop, side='right')
    X_golden = X_golden[:, :end_idx]
    t_golden = t_golden[:end_idx]
    golden_final = X_golden[:, -1]

    _, integral_golden = compute_total_accel(X_golden[1::2, :], t_golden)
    print(f"Golden reference total accel integral: {integral_golden:.2f}")

    # --- Run Comparisons ---
    q_targets = [195, 190, 175, 150]
    
    # 1. Run FULL Model once as Ground Truth
    print("\nSimulating FULL Model (Ground Truth)...")
    tracemalloc.start()
    start = time.time()
    x_full, t_full = trapezoidal(
        eval_f=lambda x, _, u: A @ x + B * u, 
        x_start=x0, p=param_dict, eval_u=eval_u,
        t_start=t_start, t_stop=t_stop, timestep=0.005,
        errf=1e-8, errDeltax=1e-8, relDeltax=1e-8, MaxIter=50, FiniteDifference=0,
        Jf_linear=A, Jf_eval_nonlinear=None, use_tqdm=True, 
        newton_linear_solver="LU"
    )
    y_full, integral_full = compute_total_accel(x_full[1::2, :], t_full)
    end = time.time()
    print(f"FULL Model simulation took {end - start:.2f}s")
    current, peak_integration = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory during FULL model integration: {peak_integration / (1024 * 1024):.2f} MB")

    # 2. Run Reduced Models
    results = {}
    
    for q in q_targets:
        results[q] = {}
        print(f"\n--- Reducing to q={q} ---")
        
        # Pass x0 to fix the misalignment issue
        tracemalloc.start()
        start = time.time()
        T_eig, A_eig, B_eig, _ = eigenmode_truncation(A, B, C_full, q, x0=x0)
        end = time.time()
        print(f"Eigenmode Truncation Reduction to q={q} took {end - start:.2f}s")
        current, peak_generation = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Peak memory during reduction: {peak_generation / (1024 * 1024):.2f} MB")
        x0_eig = T_eig.T @ x0

        # Simulate the reduced system
        print(f"Simulating Eigenmode Truncation Reduced System (q={q})...")
        tracemalloc.start()
        start = time.time()
        x_e, t_e = trapezoidal(
            eval_f=lambda x, _, u: A_eig @ x + (B_eig.flatten() * u),
            x_start=x0_eig, p=param_dict, eval_u=eval_u,
            t_start=t_start, t_stop=t_stop, timestep=0.005,
            errf=1e-8, errDeltax=1e-8, relDeltax=1e-8, MaxIter=50, FiniteDifference=0,
            Jf_linear=A_eig, Jf_eval_nonlinear=None, use_tqdm=False, 
            newton_linear_solver="LU"
        )
        end = time.time()
        current, peak_integration = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Eigenmode Truncation simulation (q={q}) took {end - start:.2f}s")
        print(f"Peak memory during integration: {peak_integration / (1024 * 1024):.2f} MB")
        x_e = T_eig @ x_e  # Lift back to full space for comparison
        results[q] = (t_e, *compute_total_accel(x_e[1::2, :], t_e), x_e)

    # Plot the total accel over time
    fig, axes = plt.subplots(len(q_targets), 1, figsize=(10, 5 * len(q_targets)), sharex=True)
    if len(q_targets) == 1: axes = [axes]

    for i, q in enumerate(q_targets):
        ax = axes[i]
        # Plot Full
        ax.plot(t_full, y_full, 'k-', linewidth=3, alpha=0.3, label='Full Model (N=200)')
        
        # Plot Reduced
        t_e, y_e, integral_e, _ = results[q]
        ax.plot(t_e, y_e, 'r--', linewidth=1.5, label=f'Eigen Truncation (q={q})')
        
        ax.set_ylabel('Total Accel (m/s²)')
        ax.set_title(f'Comparison at Reduced Order q={q}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"q={q}: Integral Full={integral_full:.2f}, Integral Eigen={integral_e:.2f}")
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    # Plot the car positions over time
    fig, axes = plt.subplots(len(q_targets), 1, figsize=(10, 5 * len(q_targets)), sharex=True)
    if len(q_targets) == 1: axes = [axes]

    for i, q in enumerate(q_targets):
        ax = axes[i]

        # Plot Full
        for car_idx in range(0, n_cars):
            ax.plot(t_full, x_full[2*car_idx, :], 'k-', linewidth=2, alpha=0.3)

        # Plot Reduced
        t_e, _, _, x_e = results[q]
        for car_idx in range(0, n_cars):
            ax.plot(t_e, x_e[2*car_idx, :], 'r--', linewidth=1.5)

        ax.set_ylabel('Position (m)')
        ax.set_title(f'Car Positions at Reduced Order q={q}')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    # Plot the car velocities over time
    fig, axes = plt.subplots(len(q_targets), 1, figsize=(10, 5 * len(q_targets)), sharex=True)
    if len(q_targets) == 1: axes = [axes]

    for i, q in enumerate(q_targets):
        ax = axes[i]

        # Plot Full
        for car_idx in range(0, n_cars):
            ax.plot(t_full, x_full[2*car_idx + 1, :], 'k-', linewidth=2, alpha=0.3)

        # Plot Reduced
        t_e, _, _, x_e = results[q]
        for car_idx in range(0, n_cars):
            ax.plot(t_e, x_e[2*car_idx + 1, :], 'r--', linewidth=1.5)

        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Car Velocities at Reduced Order q={q}')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    # Print the errors of each method's final state from the golden reference
    print("\nFinal State Errors from Golden Reference:")
    final_full = x_full[:, -1]
    error_full = np.max(np.abs(final_full - golden_final))
    print(f"  FULL Model Final State Error: {error_full:.6e} (Confidence: {confidence_golden:.6e})")

    for q in q_targets:
        _, _, _, x_e = results[q]
        final_eig = x_e[:, -1]
        error_eig = np.max(np.abs(final_eig - golden_final))
        print(f"  q={q} Eigen Truncation Final State Error: {error_eig:.6e} (Confidence: {confidence_golden:.6e})")

    # Print the errors of each method's total accel integral from the golden reference
    print("\nTotal Accel Integral Errors from Golden Reference:")
    error_integral_full = np.abs(integral_full - integral_golden)
    print(f"  FULL Model Total Accel Integral Error: {error_integral_full:.6e}")

    for q in q_targets:
        _, _, integral_e, _ = results[q]
        error_integral_eig = np.abs(integral_e - integral_golden)
        print(f"  q={q} Eigen Truncation Total Accel Integral Error: {error_integral_eig:.6e}")

if __name__ == "__main__":
    perform_eigmode_truncation_reduction(seed=42)