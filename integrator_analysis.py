import numpy as np
import time
import matplotlib.pyplot as plt

from our_solvers.trapezoidal import trapezoidal
from our_solvers.trapezoidal_adaptive import trapezoidal_adaptive
from provided_solvers.SimpleSolver import SimpleSolver
from evaluate_f import eval_f, Parameters
from evaluate_Jf import eval_Jf
import os

ACCEPTABLE_ERROR = 1e-3
"""
This is the acceptable error threshold for our traffic simulation problem. 
We choose 1mm, as any smaller error is negligible in the context of vehicles whose
lengths are on the order of meters.
"""


if __name__=="__main__":
    # Turn off multithreading for consistent timing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Set up a problem with N cars with varying parameters
    np.random.seed(42)
    num_cars = 300
    alpha_mean, alpha_std = 1.0, 0.1
    beta_mean, beta_std = 1.0, 0.1
    tau_mean, tau_std = 1.0, 0.2
    K_mean, K_std = 1.0, 0.5
    L_mean, L_std = 0.5, 0.1
    d0_mean, d0_std = 2.0, 0.5

    parameters = []
    for i in range(num_cars):
        alpha = np.random.normal(alpha_mean, alpha_std)
        beta = np.random.normal(beta_mean, beta_std)
        tau = np.random.normal(tau_mean, tau_std)
        K = np.random.normal(K_mean, K_std)
        L = np.random.normal(L_mean, L_std)
        d0 = np.random.normal(d0_mean, d0_std)
        parameters.append(Parameters(alpha, beta, tau, K, L, d0))

    parameters = {"parameters": parameters}

    # Initial state: randomly initialize positions and velocities
    x0 = np.zeros(2 * num_cars)
    v_mean, v_std = 3.0, 2.0
    d_mean, d_std = 10.0, 3.0

    for i in range(num_cars):
        x0[2*i] = np.clip(np.random.normal(d_mean, d_std), 0, None) + x0[2*(i-1)] if i > 0 else 0  # position
        x0[2*i + 1] = np.clip(np.random.normal(v_mean, v_std), 0, None)  # velocity
    
    # Time integration parameters
    t_start = 0.0
    t_stop = 1.0

    # First, compute the golden reference
    golden_reference_timestep = 1e-4

    if not os.path.exists("references/golden_reference_integrator_analysis.npz"):
        print("Computing golden reference solution...")
        
        # Simulate golden reference
        X_golden, t_golden = SimpleSolver(
            eval_f,
            x0,
            parameters,
            eval_u=lambda t: None,
            NumIter=int((t_stop - t_start) / golden_reference_timestep),
            w=golden_reference_timestep,
            use_tqdm=True,
            visualize=False
        )

        # Simulate trajectory with slightly larger timestep to quantify confidence
        X_check, t_check = SimpleSolver(
            eval_f,
            x0,
            parameters,
            eval_u=lambda t: None,
            NumIter=int((t_stop - t_start) / (golden_reference_timestep * 2)),
            w=golden_reference_timestep * 2,
            use_tqdm=True,
            visualize=False
        )

        # Compute the error between the two solutions at final time
        confidence_golden = np.linalg.norm(X_golden[:, -1] - X_check[:, -1], np.inf)
        print(f"Final error between golden reference and check solution: {confidence_golden:.6e}")
        
        if confidence_golden > ACCEPTABLE_ERROR:
            print("Warning: The error between the golden reference and check solution exceeds the acceptable threshold.")

        # Save golden reference to disk for reuse
        base_dir = os.path.dirname(__file__)
        golden_path = os.path.join(base_dir, "references/golden_reference_integrator_analysis.npz")

        np.savez_compressed(
            golden_path,
            X_golden=X_golden,
            t_golden=t_golden,
            num_cars=num_cars,
            t_start=t_start,
            t_stop=t_stop,
            golden_timestep=golden_reference_timestep,
            x0=x0,
            confidence=confidence_golden,
        )
        print(f"Golden reference saved to: {golden_path}")
    else:
        # Load golden reference from disk
        try:
            base_dir = os.path.dirname(__file__)
        except NameError:
            base_dir = os.getcwd()
        golden_path = os.path.join(base_dir, "references/golden_reference_integrator_analysis.npz")

        data = np.load(golden_path)
        X_golden = data['X_golden']
        t_golden = data['t_golden']
        confidence_golden = data['confidence']
        print(f"Golden reference loaded from: {golden_path}")
    golden_final = X_golden[:, -1]

    # Find largest delta t for forward euler before it becomes unstable
    # NOTE: This timestep is stable, and its error exceeds the acceptable (eps_a < eps_unst)
    unstable_euler_timestep = 0.1
    unstable_euler_X, unstable_euler_t = SimpleSolver(
        eval_f, 
        x0, 
        parameters, 
        eval_u=lambda t: None, 
        NumIter=int(t_stop / unstable_euler_timestep), 
        w=unstable_euler_timestep, 
        use_tqdm=True, 
        visualize=False
    )
    
    unstable_euler_error = np.linalg.norm(golden_final - unstable_euler_X[:, -1], np.inf)
    print(f"Unstable boundary for forward euler at timestep {unstable_euler_timestep} with error {unstable_euler_error}.")

    # Find the delta t that gives us error below the acceptable error.
    simple_start_time = time.time()
    # simple_timestep = 2.8e-4
    simple_timestep = 0.1
    simple_X, simple_t = SimpleSolver(
        eval_f, 
        x0, 
        parameters, 
        eval_u=lambda t: None, 
        NumIter=int(t_stop / simple_timestep), 
        w=simple_timestep, 
        use_tqdm=True, 
        visualize=False
    )
    simple_end_time = time.time()
    print(f"SimpleSolver integration completed in {simple_end_time - simple_start_time:.2f} seconds.")

    # Integrate using standard trapezoidal
    trap_start_time = time.time()
    trap_timestep = 0.5e-1
    trap_X, trap_t = trapezoidal(
        eval_f=eval_f,
        x_start=x0,
        p=parameters,
        eval_u=lambda t: None,
        t_start=t_start,
        t_stop=t_stop,
        timestep=trap_timestep,
        errf=1e-8,
        errDeltax=1e-8,
        relDeltax=1e-8,
        MaxIter=50,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        use_tqdm=True,
        newton_linear_solver="LU",
    )
    trap_end_time = time.time()

    print(f"Standard trapezoidal integration completed in {trap_end_time - trap_start_time:.2f} seconds.")

    # Integrate using trapezoidal with solve_banded linear solver
    trap_banded_start_time = time.time()
    trap_banded_X, trap_banded_t = trapezoidal(
        eval_f=eval_f,
        x_start=x0,
        p=parameters,
        eval_u=lambda t: None,
        t_start=t_start,
        t_stop=t_stop,
        timestep=trap_timestep,
        errf=1e-8,
        errDeltax=1e-8,
        relDeltax=1e-8,
        MaxIter=50,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        use_tqdm=True,
        newton_linear_solver="solve_banded",
        Jf_bandwidth=(1, 2),
    )
    trap_banded_end_time = time.time()

    print(f"Solve_banded trapezoidal integration completed in {trap_banded_end_time - trap_banded_start_time:.2f} seconds.")

    # Integrate using trapezoidal with TGCR linear solver
    trap_tgcr_start_time = time.time()
    trap_tgcr_X, trap_tgcr_t = trapezoidal(
        eval_f=eval_f,
        x_start=x0,
        p=parameters,
        eval_u=lambda t: None,
        t_start=t_start,
        t_stop=t_stop,
        timestep=trap_timestep,
        errf=1e-8,
        errDeltax=1e-8,
        relDeltax=1e-8,
        MaxIter=50,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        use_tqdm=True,
        newton_linear_solver="tgcr",
    )
    trap_tgcr_end_time = time.time()

    print(f"TGCR trapezoidal integration completed in {trap_tgcr_end_time - trap_tgcr_start_time:.2f} seconds.")

    # Integratue using trapezoidal with solve_banded linear solver and adaptive timestep
    trap_adaptive_start_time = time.time()
    trap_adaptive_X, trap_adaptive_t = trapezoidal_adaptive(
        eval_f=eval_f,
        x_start=x0,
        p=parameters,
        eval_u=lambda t: None,
        t_start=t_start,
        t_stop=t_stop,
        initial_timestep=trap_timestep,
        min_step_size=trap_timestep,
        errf=1e-8,
        errDeltax=1e-8,
        relDeltax=1e-8,
        MaxIter=50,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        use_tqdm=True,
        newton_linear_solver="solve_banded",
        return_full_traj=True,
        Jf_bandwidth=(1, 2),
    )
    trap_adaptive_end_time = time.time()

    print(f"Adaptive trapezoidal integration completed in {trap_adaptive_end_time - trap_adaptive_start_time:.2f} seconds.")

    # Compute and print errors against the golden reference at final time
    trap_final = trap_X[:, -1]
    trap_banded_final = trap_banded_X[:, -1]
    trap_tgcr_final = trap_tgcr_X[:, -1]
    simple_final = simple_X[:, -1]

    trap_error = np.linalg.norm(trap_final - golden_final, np.inf)
    trap_banded_error = np.linalg.norm(trap_banded_final - golden_final, np.inf)
    trap_tgcr_error = np.linalg.norm(trap_tgcr_final - golden_final, np.inf)
    trap_adaptive_error = np.linalg.norm(trap_adaptive_X[:, -1] - golden_final, np.inf)
    simple_error = np.linalg.norm(simple_final - golden_final, np.inf)

    print(f"Golden reference confidence level: {confidence_golden:.6e}")
    print(f"Acceptable forward euler max error at final time: {simple_error:.6e}")
    print(f"Standard trapezoidal max error at final time: {trap_error:.6e}")
    print(f"Solve_banded trapezoidal max error at final time: {trap_banded_error:.6e}")
    print(f"TGCR trapezoidal max error at final time: {trap_tgcr_error:.6e}")
    print(f"Adaptive trapezoidal max error at final time: {trap_adaptive_error:.6e}")

    # Plot the trajectories of the last 5 cars for the golden reference.
    plt.figure(figsize=(12, 8))
    for i in range(num_cars-5, num_cars):
        plt.plot(t_golden, X_golden[2*i, :], label=f'Golden Car {i+1}', linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Trajectories of the last 5 cars - Golden Reference')
    plt.legend()
    plt.show()

    # Plot the trajectories of the last 5 cars for the unstable forward euler.
    plt.figure(figsize=(12, 8))
    for i in range(num_cars-5, num_cars):
        plt.plot(t_golden, X_golden[2*i, :], label=f'Golden Car {i+1}', linestyle=':')
        plt.plot(unstable_euler_t, unstable_euler_X[2*i, :], label=f'Unstable Euler Car {i+1}', marker='x')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Trajectories of the last 5 cars - Unstable Forward Euler')
    plt.legend()
    plt.show()

    # Plot the trajectories of the last 5 cars for acceptable forward euler.
    plt.figure(figsize=(12, 8))
    for i in range(num_cars-5, num_cars):
        plt.plot(t_golden, X_golden[2*i, :], label=f'Golden Car {i+1}', linestyle=':')
        plt.plot(simple_t, simple_X[2*i, :], label=f'Acceptable Euler Car {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Trajectories of the last 5 cars - Acceptable Forward Euler')
    plt.legend()
    plt.show()

    # Plot the trajectories of the last 5 cars for trapezoidal solve_banded and adaptive methods.
    plt.figure(figsize=(12, 8))
    for i in range(num_cars-5, num_cars):
        plt.plot(t_golden, X_golden[2*i, :], label=f'Golden Car {i+1}', linestyle=':')
        plt.plot(trap_banded_t, trap_banded_X[2*i, :], label=f'Trapezoidal Solve_banded Car {i+1}', marker='o')
        plt.plot(trap_adaptive_t, trap_adaptive_X[2*i, :], label=f'Trapezoidal Adaptive Car {i+1}', marker='s')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Trajectories of the last 5 cars')
    plt.legend()
    plt.show()

    # Plot the timestep history for the adaptive trapezoidal method
    # Put a horizontal line at 0.05s, which is the fixed timestep used in other trapezoidal methods
    plt.figure(figsize=(12, 6))
    plt.plot(trap_adaptive_t[:-1], np.diff(trap_adaptive_t), marker='o')
    plt.axhline(y=trap_timestep, color='r', linestyle='--', label='Fixed Timestep (0.05s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Timestep (s)')
    plt.title('Adaptive Trapezoidal Method Timestep History')
    plt.show()