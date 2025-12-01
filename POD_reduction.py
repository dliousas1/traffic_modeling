import numpy as np
import matplotlib.pyplot as plt
from evaluate_f import Parameters
from evaluate_Jf import eval_Jf_analytic_linear as eval_Jf
from provided_solvers.reduce_system import reduce_system
from provided_solvers.eval_f_LinearSystem import eval_f_LinearSystem
from provided_solvers.eval_Jf_LinearSystem import eval_Jf_LinearSystem
from provided_solvers.implicit import implicit
from provided_solvers.eval_u_cos import gen_eval_u_cos
from icecream import ic
import datetime
import numpy as np
import time 

if __name__ == "__main__":
    n = 100
    parameters = [Parameters(2.0, 0.5, 0.5, 0.0, 0.0, 0.0) for _ in range(n)]
    param_dict = {"parameters": parameters}
    dx = 15.0
    initial_positions = [(i * dx) for i in range(n)]
    initial_velocities = [20.0 for _ in range(n)]

    initial_conditions = []
    for i in range(n):
        initial_conditions.append(initial_positions[i])
        initial_conditions.append(initial_velocities[i])
    initial_conditions = np.array(initial_conditions)

    A = eval_Jf(initial_conditions, param_dict, n)
    # Make B matrix so input only affects velocity of leading vehicle (nth vehicle)
    B = np.zeros((2 * n, 1))
    B[2 * n - 1, 0] = 3.0

    # Make C matrix to observe the average velocity of all vehicles
    C = np.zeros((1, 2 * n))
    for i in range(n):
        C[0, 2 * i + 1] = 1.0 / n

    q_vals = [100, 75, 50, 25]  # Reduced order
    method = 'POD'  # Reduction method

    training_data = None  # To store training data for plotting later
    filenames = []
    for q in q_vals:
        print(f"\nPerforming POD reduction to order q={q}...")
        Vq, Ar, Br, Cr, xr_start, extra_outputs = reduce_system(A, B, C, initial_conditions, q, method, training_data=training_data)

        # Unpack extra outputs
        X_training, U, S, duration_training, duration_svd, p, t_start, t_stop, dt = extra_outputs

        # Save data with time stamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'POD_reduction_data_{timestamp}_q{q}.npz'
        np.savez(filename,
                Vq=Vq, Ar=Ar, Br=Br, Cr=Cr, xr_start=xr_start,
                X_training=X_training, U=U, S=S,
                duration_training=duration_training, duration_svd=duration_svd,
                p=p, t_start=t_start, t_stop=t_stop, dt=dt)
        filenames.append(filename)
        training_data = (X_training, p, t_start, t_stop, dt, duration_training)  # For next iteration

    # Load the reference solution from POD_reduction_data_20251130_225734_ref_dt1e-3.npz
    ref_data = np.load('POD_reduction_data_20251130_225734_ref_dt1e-3.npz', allow_pickle=True)
    X_training_ref = ref_data['X_training']
    U_ref = ref_data['U']
    S_ref = ref_data['S']
    t_start_ref = ref_data['t_start']
    t_stop_ref = ref_data['t_stop']
    dt_ref = ref_data['dt'] 

    def get_relative_position_states(X):
        """
        Assume X has shape (states, times) with states ordered as
        [pos0, vel0, pos1, vel1, ..., pos_{n-1}, vel_{n-1}].
        Returns positions relative to the lead car as an array of shape (n, times).
        """
        positions = X[0::2, :]           # (n, times)
        lead_pos = positions[-1, :]      # (times,)
        positions_rel = positions - lead_pos[np.newaxis, :]
        return positions_rel
    
    def compute_total_acceleration(X, t):
        """
        Compute the total acceleration experienced by all cars at each time step.
        X: state matrix of shape (states, times)
        t: time vector of shape (times,)
        Returns: total_acceleration: array of shape (times,)
        """
        accelerations = np.zeros_like(X)
        for i in range(1, X.shape[1]):
            dt_sim = t[i] - t[i - 1]
            accelerations[:, i] = (X[:, i] - X[:, i - 1]) / dt_sim
        return np.sum(np.abs(accelerations), axis=0)

    X_ref_rel = get_relative_position_states(X_training_ref)
    X_rel = get_relative_position_states(X_training)

    t_start = 0.0
    t_stop = 100.0
    dt = 0.01
    time_steps = np.arange(t_start, t_stop + dt/2, dt)

    accel_dict = {}
    accel_dict["Full Model"] = compute_total_acceleration(X_training, time_steps)

    # Compute the max absolute difference in the final time step
    # Find index where ref time matches 100 seconds
    time_steps_ref = np.arange(t_start_ref, t_stop_ref + dt_ref/2, dt_ref)
    idx_100s_ref = np.where(np.isclose(time_steps_ref, 100.0))[0][0]
    final_step_ref = X_ref_rel[:, idx_100s_ref]
    final_step = X_ref_rel[:, -1]

    max_abs_diff = np.max(np.abs(final_step - final_step_ref))
    print(f"\nMax absolute difference in final time step compared to reference (CONFIDENCE): {max_abs_diff:.6e}")


    # Compute the reduced order time simulation for each q and and get the error at final time compared to trainign data
    for q, filename in zip(q_vals, filenames):
        data = np.load(filename, allow_pickle=True)
        Vq = data['Vq']
        Ar = data['Ar']
        Br = data['Br']
        Cr = data['Cr']
        xr_start = data['xr_start']

        p = {'A': Ar, 'B': Br}
        eval_f = eval_f_LinearSystem
        eval_Jf = eval_Jf_LinearSystem
        T = 5.0
        eval_u = gen_eval_u_cos(T)  # Cosine input for training
        p['T'] = T
        # Initial conditions
        t_start = 0

        t_stop = 100.0
        dt = 0.01
        visualize = False
        finite_difference = False
        
        t_integration_start = time.time()
        # Use implicit integration (Trapezoidal method)
        Xr, t_r, _ = implicit(
            'Trapezoidal', eval_f, xr_start, p, eval_u, t_start, t_stop, dt, visualize, finite_difference, eval_Jf
        )
        t_integration_end = time.time()
        duration_integration = t_integration_end - t_integration_start
        print(f"Integrated reduced system of order q={q} in {duration_integration:.2f} seconds.")
        

        # Project back to full space
        X_full_approx = Vq @ Xr

        X_full_approx_rel = get_relative_position_states(X_full_approx)

        # Compute error at final time step
        final_step_approx = X_full_approx_rel[:, -1]
        max_abs_diff_approx = np.max(np.abs(final_step_approx - final_step_ref))
        print(f"Max absolute difference in final time step for q={q}: {max_abs_diff_approx:.6e}")

        # Compute total acceleration for reduced model
        accel_approx = compute_total_acceleration(X_full_approx, time_steps)
        accel_dict[f"q={q}"] = accel_approx

    # Plot training data results
    x = X_training
    # Plot the training data positions and velocities
    t = np.arange(t_start, t_stop + dt/2, dt)

    pos_lead = initial_positions[-1] + initial_velocities[-1] * t
    # Plot the results on subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    plot_rel_pos = True
    plot_rel_vel = False
    # Plot positions of the cars over time relative to leading car
    for i in range(n):
        if plot_rel_pos:
            axs[0].plot(t, x[2*i, :] - x[2*(n-1), :], label=f'Position car {i+1} relative to car {n}')
            # axs[0].plot(t, x[2*i, :] - pos_lead, label=f'Position car {i+1} relative to unperturbed leading car')
            # Position relative to car in front
            # if i < n - 1:
            #     axs[0].plot(t, x[2*i, :] - x[2*(i+1), :], label=f'Position car {i+1} relative to car {i+2}')
        else:
            axs[0].plot(t, x[2*i, :], label=f'Position car {i+1}')
    # axs[0].plot(t, pos_lead, 'k--', label='Unperturbed leading car')
    axs[0].set_xlabel('Time (s)')
    if plot_rel_pos:
        axs[0].set_ylabel('Relative Position (m)')
        axs[0].set_title('Relative Position vs Time')
    else:
        axs[0].set_ylabel('Position (m)')
        axs[0].set_title('Position vs Time')
    # axs[0].legend()

    # Plot velocities of the cars over time
    for i in range(n):
        if plot_rel_vel:
            axs[1].plot(t, x[2*i + 1, :] - x[2*(n-1) + 1, :], label=f'Velocity car {i+1} relative to car {n}')
        else:
            axs[1].plot(t, x[2*i + 1, :], label=f'Velocity car {i+1}')

    axs[1].set_xlabel('Time (s)')

    if plot_rel_vel:
        axs[1].set_ylabel('Relative Velocity (m/s)')
        axs[1].set_title('Relative Velocity vs Time')
    else:
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].set_title('Velocity vs Time')

    # axs[1].legend()
    plt.tight_layout()

    # Plot the singular values
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(S) + 1), S, 'o-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Values from POD Training Data')

    # Plot total acceleration comparison for full and reduced models
    plt.figure(figsize=(8, 5))
    for key, accel in accel_dict.items():
        plt.plot(time_steps, accel, label=key)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Acceleration')
    plt.title('Total Acceleration Comparison')
    plt.legend()
    # Save plot as png and pdf
    plt.savefig('total_acceleration_comparison.png', dpi=300)
    plt.savefig('total_acceleration_comparison.pdf')


    plt.show()



