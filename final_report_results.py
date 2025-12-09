import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Literal
import os

from our_solvers.trapezoidal_adaptive import trapezoidal_adaptive
from our_solvers.trapezoidal import trapezoidal
from provided_solvers.SimpleSolver import SimpleSolver
from evaluate_f import eval_f, Parameters
from evaluate_Jf import eval_Jf
from animate_traffic import animate_traffic

def compute_total_accel(velocities, t):
    accelerations = np.zeros_like(velocities)
    for i in range(1, velocities.shape[1]):
        dt_sim = t[i] - t[i - 1]
        accelerations[:, i] = (velocities[:, i] - velocities[:, i - 1]) / dt_sim

    accelerations = np.sum(np.abs(accelerations), axis=0)
    accel_integral = np.trapezoid(accelerations, t)

    return accelerations, accel_integral

def simulate_phantom_jam(driver_type: Literal["all unsafe", "all safe", "one safe"], animate: bool = False):
    # Define the number of cars in the example
    n_cars = 10

    # Define the parameters and initial states for each car
    param_dict = {"parameters": []}
    x0 = []
    for i in range(n_cars):
        if driver_type == "all unsafe":
            tau = 0.2  # Unsafe drivers
            starting_dist = 3.0

        elif driver_type == "all safe":
            tau = 0.5  # Safe drivers
            starting_dist = 7.5
        
        elif driver_type == "one safe":
            if i == n_cars - 4:
                tau = 0.7  # Safe driver
            else:
                tau = 0.2  # Unsafe drivers
            starting_dist = 3.0
        else:
            raise ValueError("Invalid driver_type. Choose from 'all unsafe', 'all safe', 'one safe'.")
        
        # Define parameters for each car (example)
        params = Parameters(
            alpha=1.0,
            beta=1.0,
            tau=tau,
            K=3.0,
            d0=2.0,
            L=2.0,
        )
        param_dict["parameters"].append(params)

        # Define initial state for each car (position and velocity)
        if driver_type == "one safe" and i > n_cars - 4:
            position = i * starting_dist + 7.5
        else:
            position = i * starting_dist
        velocity = 15.0
        x0.extend([position, velocity])

    # Modify the second last car's position to be closer to the last car to induce phantom traffic jam
    x0[-4] = x0[-2] - 0.5
    x0 = np.array(x0)

    # Simulation parameters
    t0 = 0.0
    tf = 20.0
    rtol = 1e-6
    atol = 1e-8
    errf = 1e-8

    # Run the simulation using the trapezoidal adaptive solver
    x, t = trapezoidal_adaptive(
        eval_f=eval_f,
        x_start=x0,
        p=param_dict,
        eval_u=lambda t: 0.0,
        t_start=t0,
        t_stop=tf,
        initial_timestep=0.01,
        errf=errf,
        errDeltax=atol,
        relDeltax=rtol,
        MaxIter=100000,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        use_tqdm=True,
        min_step_size=0.0001,
        newton_linear_solver="solve_banded",
        Jf_bandwidth=(1, 2),
    )

    if animate:
        positions = x[0::2, :].T
        animate_traffic(positions, t)

    # Plot the positions of the cars over time
    plt.figure(figsize=(12, 6))
    for i in range(n_cars):
        plt.plot(t, x[2*i, :], label=f'Car {i+1} Position')
    plt.title('Traffic Flow Simulation: Car Positions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(loc="lower right")
    plt.grid()

    # Plot the velocities of the cars over time
    plt.figure(figsize=(12, 6))
    for i in range(n_cars):
        plt.plot(t, x[2*i + 1, :], label=f'Car {i+1} Velocity')
    plt.title('Traffic Flow Simulation: Car Velocities Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid()

    # Plot the total acceleration over time
    accelerations, accel_integral = compute_total_accel(x[1::2, :], t)
    plt.figure(figsize=(12, 6))
    plt.plot(t, accelerations, label='Total Acceleration')
    plt.title('Total Acceleration Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Acceleration (m/s²)')
    plt.legend()
    plt.grid()
    plt.show()

    print(f'Total Acceleration Integral over time: {accel_integral}')

def sweep_safety_parameter():
    taus = np.linspace(0.1, 1.0, 10)
    velocity = 15.0

    def solve_equilibrium_distance_new_params(tau, v, tol=1e-8, max_iter=20):
        """
        Implicit solver for the equilibrium distance between two cars for a
        given safety parameter tau and velocity v.
        """
        # Constants
        K = 10.0
        L = 2.0
        d0 = 2.0
        
        # Initial Guess:
        # Since K is large, the gap will always be at least d0 + small amount.
        # We guess slightly above d0 to avoid huge exponential values in the first step.
        dz = max(d0 + 0.5, v * tau) 
        
        for k in range(max_iter):
            # 1. Compute term common to function and derivative
            # exp(-2 * (z - 2))
            dist_from_jam = dz - d0
            exp_term = np.exp(-L * dist_from_jam)
            
            # 2. Compute Residual
            # z - 10*exp(...) - v*tau
            interaction = K * exp_term
            residual = dz - interaction - (v * tau)
            
            # Check convergence
            if abs(residual) < tol:
                return dz, k
                
            # 3. Compute Jacobian
            # 1 + K * L * exp(...) = 1 + 20 * exp(...)
            jacobian = 1.0 + (K * L * exp_term)
            
            # 4. Newton Step
            dz = dz - (residual / jacobian)
            
        print(f"Warning: Did not converge for tau={tau}, v={v}")
        return dz, max_iter
    
    # Compute equilibrium (starting) distances for each tau
    eq_distances = []
    for tau in taus:
        dz, iters = solve_equilibrium_distance_new_params(tau, velocity)
        eq_distances.append(dz)

    # For each tau, simulate 10s and compute total acceleration
    total_accels = []
    for idx, tau in enumerate(taus):
        # Define parameters and initial states
        param_dict = {"parameters": []}
        x0 = []
        n_cars = 10

        np.random.seed(42)  # For reproducibility

        for i in range(n_cars):
            params = Parameters(
                alpha=1.0,
                beta=1.0,
                tau=tau + 0.05 * np.random.randn(),  # Small random variation
                K=10.0,
                d0=2.0,
                L=2.0,
            )

            if i == n_cars - 1:
                params.input_ampl = 1.0  # Enable input for the lead car only

            param_dict["parameters"].append(params)

            position = i * eq_distances[idx]
            x0.extend([position, velocity])

        x0 = np.array(x0)

        # Run simulation
        t0 = 0.0
        tf = 10.0
        rtol = 1e-6
        atol = 1e-8
        errf = 1e-8

        x, t = trapezoidal_adaptive(
            eval_f=eval_f,
            x_start=x0,
            p=param_dict,
            eval_u=lambda t: 10.0*np.sin(2*t),
            t_start=t0,
            t_stop=tf,
            initial_timestep=0.01,
            errf=errf,
            errDeltax=atol,
            relDeltax=rtol,
            MaxIter=100000,
            FiniteDifference=0,
            Jf_eval=eval_Jf,
            use_tqdm=True,
            min_step_size=0.0001,
            newton_linear_solver="solve_banded",
            Jf_bandwidth=(1, 2),
        )

        accelerations, accel_integral = compute_total_accel(x[1::2, :], t)
        total_accels.append(accel_integral)
        print(f"tau: {tau:.3f}, Total Acceleration Integral: {accel_integral}")

        # Plot the positions of the cars over time for this tau
        plt.figure(figsize=(12, 6))
        for i in range(n_cars):
            plt.plot(t, x[2*i, :], label=f'Car {i+1} Position')
        plt.title(f'Traffic Flow Simulation (tau={tau:.2f}): Car Positions Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid()
        plt.show()

    # Plot the total accel integral vs tau as a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(taus, total_accels, width=0.05)
    plt.title('Total Acceleration Integral vs Safety Parameter (tau)')
    plt.xlabel('Safety Parameter (tau)')
    plt.ylabel('Total Acceleration Integral (m/s²·s)')
    plt.grid()
    plt.show()

def perform_complexity_analysis():
    num_cars = np.arange(10, 300, 10)
    computation_times = []
    
    for n in num_cars:
        # Setup parameters and initial conditions
        param_dict = {"parameters": []}
        x0 = []
        for i in range(n):
            params = Parameters(
                alpha=1.0,
                beta=1.0,
                tau=0.5,
                K=10.0,
                d0=2.0,
                L=2.0,
            )
            param_dict["parameters"].append(params)
            position = i * 7.5
            velocity = 15.0
            x0.extend([position, velocity])
        x0 = np.array(x0)

        # Timing the simulation
        start_time = time.time()
        
        x, t = trapezoidal_adaptive(
            eval_f=eval_f,
            x_start=x0,
            p=param_dict,
            eval_u=lambda t: 0.0,
            t_start=0.0,
            t_stop=50.0,
            initial_timestep=0.01,
            errf=1e-8,
            errDeltax=1e-8,
            relDeltax=1e-6,
            MaxIter=100000,
            FiniteDifference=0,
            Jf_eval=eval_Jf,
            use_tqdm=False,
            min_step_size=0.001,
            newton_linear_solver="solve_banded",
            Jf_bandwidth=(1, 2),
        )
        
        end_time = time.time()
        computation_times.append(end_time - start_time)
        print(f"Number of Cars: {n}, Computation Time: {end_time - start_time:.2f} seconds")

    # Plot with a linear fit plotted over the data
    coeffs_linear = np.polyfit(num_cars, computation_times, 1)
    poly_linear = np.poly1d(coeffs_linear)
    fit_times_linear = poly_linear(num_cars)
    plt.figure(figsize=(10, 6))
    plt.plot(num_cars, computation_times, marker='o', label='Measured Times')
    plt.plot(num_cars, fit_times_linear, label='Linear Fit', linestyle='--')
    plt.title('Computation Time vs Number of Cars with Linear Fit')
    plt.xlabel('Number of Cars')
    plt.ylabel('Computation Time (seconds)')
    plt.legend()
    plt.grid()
    plt.show()

def demo_adaptive_timestepper():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Define the number of cars in the example
    n_cars = 50

    np.random.seed(42)
    alpha_mean, alpha_std = 1.0, 0.1
    beta_mean, beta_std = 1.0, 0.1
    tau_mean, tau_std = 1.0, 0.2
    K_mean, K_std = 1.0, 0.5
    L_mean, L_std = 0.5, 0.1
    d0_mean, d0_std = 2.0, 0.5

    parameters = []
    for i in range(n_cars):
        alpha = np.random.normal(alpha_mean, alpha_std)
        beta = np.random.normal(beta_mean, beta_std)
        tau = np.random.normal(tau_mean, tau_std)
        K = np.random.normal(K_mean, K_std)
        L = np.random.normal(L_mean, L_std)
        d0 = np.random.normal(d0_mean, d0_std)
        parameters.append(Parameters(alpha, beta, tau, K, L, d0))

    param_dict = {"parameters": parameters}

    # Initial state: randomly initialize positions and velocities
    x0 = np.zeros(2 * n_cars)
    v_mean, v_std = 3.0, 2.0
    d_mean, d_std = 10.0, 3.0

    for i in range(n_cars):
        x0[2*i] = np.clip(np.random.normal(d_mean, d_std), 0, None) + x0[2*(i-1)] if i > 0 else 0  # position
        x0[2*i + 1] = np.clip(np.random.normal(v_mean, v_std), 0, None)  # velocity

    # Simulation parameters
    t0 = 0.0
    tf = 100.0
    rtol = 1e-6
    atol = 1e-8
    errf = 1e-8

    # Get a "golden" solution for reference
    ref_path = "references/golden_solution_adaptive_demo.npz"
    
    if not os.path.exists(ref_path):
        x_gold, t_gold = SimpleSolver(
            eval_f=eval_f,
            x_start=x0,
            p=param_dict,
            eval_u=lambda t: 0.0,
            NumIter=int((tf - t0) / 0.0001),
            w = 0.0001,
            use_tqdm=True,
            visualize=False,
        )
        np.savez(ref_path, x_gold=x_gold, t_gold=t_gold)
    else:
        data = np.load(ref_path)
        x_gold = data['x_gold']
        t_gold = data['t_gold']

    fixed_dt = 0.05

    # Time to run adaptive vs fixed timestepper

    start = time.time()
    # Run the simulation using the trapezoidal adaptive solver
    x_adaptive, t_adaptive = trapezoidal_adaptive(
        eval_f=eval_f,
        x_start=x0,
        p=param_dict,
        eval_u=lambda t: 0.0,
        t_start=t0,
        t_stop=tf,
        initial_timestep=fixed_dt,
        min_step_size=fixed_dt,
        errf=errf,
        errDeltax=atol,
        relDeltax=rtol,
        MaxIter=100000,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        use_tqdm=True,
        newton_linear_solver="solve_banded",
        Jf_bandwidth=(1, 2),
        return_full_traj=True,
    )
    end = time.time()
    print(f"Adaptive Timestepper Time: {end - start:.2f} seconds")

    start = time.time()
    # Run the simulation using the fixed timestep trapezoidal solver
    x_fixed, t_fixed = trapezoidal(
        eval_f=eval_f,
        x_start=x0,
        p=param_dict,
        eval_u=lambda t: 0.0,
        t_start=t0,
        t_stop=tf,
        timestep=fixed_dt,
        MaxIter=100000,
        FiniteDifference=0,
        Jf_eval=eval_Jf,
        newton_linear_solver="solve_banded",
        Jf_bandwidth=(1, 2),
        use_tqdm=True,
    )
    end = time.time()
    print(f"Fixed Timestepper Time: {end - start:.2f} seconds")

    # Plot the timesteps taken by each method
    plt.figure(figsize=(10, 6))
    plt.plot(t_adaptive[:-1], np.diff(t_adaptive), label='Adaptive Timestepper', marker='o')
    plt.hlines(fixed_dt, t0, tf, colors='r', linestyles='--', label='Fixed Timestep')
    plt.title('Timestep Comparison: Adaptive vs Fixed')
    plt.xlabel('Time (s)')
    plt.ylabel('Timestep Size (s)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the positions of the cars over time for both methods
    plt.figure(figsize=(12, 6))
    for i in range(n_cars):
        plt.plot(t_adaptive, x_adaptive[2*i, :], label=f'Car {i+1} Position (Adaptive)', linestyle='-')
        plt.plot(t_fixed, x_fixed[2*i, :], label=f'Car {i+1} Position (Fixed)', linestyle='--')
    plt.title('Traffic Flow Simulation: Car Positions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid()
    plt.show()

    # Compute and print maximum error from golden solution at final time
    gold_final = x_gold[:, -1]
    adaptive_final = x_adaptive[:, -1]
    fixed_final = x_fixed[:, -1]

    adaptive_error = np.linalg.norm(adaptive_final - gold_final, np.inf)
    fixed_error = np.linalg.norm(fixed_final - gold_final, np.inf)
    
    print(f'Maximum Error at Final Time:')
    print(f'Adaptive Timestepper Error: {adaptive_error:.6e}')
    print(f'Fixed Timestepper Error: {fixed_error:.6e}')

def set_default_plot_params():
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

if __name__ == "__main__":
    set_default_plot_params()
    # simulate_phantom_jam("all unsafe", animate=True)
    # simulate_phantom_jam("all safe", animate=True)
    # simulate_phantom_jam("one safe", animate=True)
    # sweep_safety_parameter()
    # perform_complexity_analysis()
    demo_adaptive_timestepper()
    pass