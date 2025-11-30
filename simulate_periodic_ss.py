import matplotlib.pyplot as plt

from provided_solvers import forward_euler, implicit, shooting_Newton
from provided_solvers.eval_u_cos import gen_eval_u_cos

from evaluate_f import eval_f, Parameters
from evaluate_Jf import eval_Jf_analytic_linear as eval_Jf
from icecream import ic
from datetime import datetime
import numpy as np


# if __name__ == "__main__":
#     # Generate parameters and initial conditions for n cars
#     n = 100
#     parameters = [Parameters(1.0, 1.0, 1.0, 0, 1.0, 3.0) for _ in range(n)]

#     # Set the amplitude of the cosine input for the leading car
#     parameters[-1].input_ampl = 3.0

#     T = 5.0  # Period of the cosine input

#     param_dict = {"parameters": parameters, "dxFD":1e-8}
#     # Set the positions initial conditions with nth car leading
#     dx = 15.0
#     initial_positions = [(i * dx) for i in range(n)]
#     # Set all velocties to 10
#     initial_velocities = [20.0 for _ in range(n)]
    

#     initial_conditions = []
#     for i in range(n):
#         initial_conditions.append(initial_positions[i])
#         initial_conditions.append(initial_velocities[i])

#     t_start = 0.0
#     t_end = 50.0
#     # Simulate the system using Forward Euler method
#     # x, t = forward_euler.forward_euler(eval_f, initial_conditions, param_dict, gen_eval_u_cos(T), t_start, t_end, 0.0001, visualize=False)  

#     # Record time to run Implicit method
#     t_init = datetime.now()
    
#     # Simulate the system using Implicit method backward Euler
#     x, t, k = implicit.implicit("Trapezoidal", eval_f, initial_conditions, param_dict, gen_eval_u_cos(T), t_start, t_end, 0.05, False, True, use_GCR=True)
#     t_final = datetime.now()
#     total_duration = t_final - t_init
#     ic(total_duration)

#     plot_rel_pos = True
#     plot_rel_vel = False

#     # Compute unperturbed position of leading car
#     pos_lead = initial_positions[-1] + initial_velocities[-1] * t

#     # Plot the results on subplots
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8))

#     # Plot positions of the cars over time relative to leading car
#     for i in range(n):
#         if plot_rel_pos:
#             # axs[0].plot(t, x[2*i, :] - x[2*(n-1), :], label=f'Position car {i+1} relative to car {n}')
#             axs[0].plot(t, x[2*i, :] - pos_lead, label=f'Position car {i+1} relative to unperturbed leading car')
#             # Position relative to car in front
#             # if i < n - 1:
#             #     axs[0].plot(t, x[2*i, :] - x[2*(i+1), :], label=f'Position car {i+1} relative to car {i+2}')
#         else:
#             axs[0].plot(t, x[2*i, :], label=f'Position car {i+1}')
#     # axs[0].plot(t, pos_lead, 'k--', label='Unperturbed leading car')
#     axs[0].set_xlabel('Time (s)')
#     if plot_rel_pos:
#         axs[0].set_ylabel('Relative Position (m)')
#         axs[0].set_title('Relative Position vs Time')
#     else:
#         axs[0].set_ylabel('Position (m)')
#         axs[0].set_title('Position vs Time')
#     # axs[0].legend()

#     # Plot velocities of the cars over time
#     for i in range(n):
#         if plot_rel_vel:
#             axs[1].plot(t, x[2*i + 1, :] - x[2*(n-1) + 1, :], label=f'Velocity car {i+1} relative to car {n}')
#         else:
#             axs[1].plot(t, x[2*i + 1, :], label=f'Velocity car {i+1}')

#     axs[1].set_xlabel('Time (s)')

#     if plot_rel_vel:
#         axs[1].set_ylabel('Relative Velocity (m/s)')
#         axs[1].set_title('Relative Velocity vs Time')
#     else:
#         axs[1].set_ylabel('Velocity (m/s)')
#         axs[1].set_title('Velocity vs Time')

#     # axs[1].legend()

#     plt.tight_layout()
#     plt.show()

#     # Save the figure as a PNG file and a PDF file
#     fig.savefig('periodic_ss_simulation.png')
#     fig.savefig('periodic_ss_simulation.pdf')

#     # Save the simulation data to a .npz file with a timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"periodic_ss_simulation_{timestamp}.npz"
#     np.savez(filename, x=x, t=t, n=n, T=T, params=parameters[0].__dict__, input_ampl=parameters[-1].input_ampl, initial_positions=initial_positions, initial_velocities=initial_velocities, duration=total_duration.total_seconds())

# if __name__ == "__main__":
#     # Load the simulation data from a .npz file
#     data = np.load('periodic_ss_simulation_20251129_190549.npz', allow_pickle=True)
#     x = data['x']
#     t = data['t']
#     n = int(data['n'])
#     dt = np.unique(np.round(np.diff(t), decimals=2)).item()
#     T = float(data['T'])
#     parameters = data['params'].item()
#     input_ampl = float(data['input_ampl'])
#     initial_positions = data['initial_positions']
#     initial_velocities = data['initial_velocities']
#     duration = float(data['duration'])
#     ic(duration)
#     ic(dt)

#     data_ref = np.load('periodic_ss_simulation_20251129_190004.npz', allow_pickle=True)
#     x_ref = data_ref['x']
#     t_ref = data_ref['t']  
#     n_ref = int(data_ref['n'])
#     dt_ref = np.unique(np.round(np.diff(t_ref), decimals=2)).item()
#     T_ref = float(data_ref['T'])
#     parameters_ref = data_ref['params'].item()
#     input_ampl_ref = float(data_ref['input_ampl'])
#     initial_positions_ref = data_ref['initial_positions']
#     initial_velocities_ref = data_ref['initial_velocities']
#     duration_ref = float(data_ref['duration'])
#     ic(duration_ref)
#     ic(dt_ref)

#     # For each solution, find how many periods before periodic steady state is reached. Loop through the state of the system at each period and compare to the state one period earlier.
#     target_times = np.arange(t[0], t[-1] + T/2, T)
#     period_indices = np.array([np.argmin(np.abs(t - tt)) for tt in target_times], dtype=int)
#     period_indices = np.unique(period_indices)  # remove any accidental duplicates

#     period_states = x[:, period_indices]
#     # Get the position and velocity differences between each car at each period
#     period_positions = period_states[0::2, :]
#     period_velocities = period_states[1::2, :]
#     period_delta_positions = np.diff(period_positions, axis=1)
#     period_delta_velocities = np.diff(period_velocities, axis=1)
#     period_delta_states = np.vstack((period_delta_positions, period_delta_velocities))
#     steady_state_periods = 0
#     diffs = np.diff(period_delta_states, axis=1)
#     max_diffs = np.max(np.abs(diffs), axis=0)



#     # Reference 
#     target_times_ref = np.arange(t_ref[0], t_ref[-1] + T_ref/2, T_ref)
#     period_indices_ref = np.array([np.argmin(np.abs(t_ref - tt)) for tt in target_times_ref], dtype=int)
#     period_indices_ref = np.unique(period_indices_ref)  # remove any accidental duplicates
#     period_states_ref = x_ref[:, period_indices_ref]
#     # Get the position and velocity differences between each car at each period
#     period_positions_ref = period_states_ref[0::2, :]
#     period_velocities_ref = period_states_ref[1::2, :]
#     period_delta_positions_ref = np.diff(period_positions_ref, axis=1)
#     period_delta_velocities_ref = np.diff(period_velocities_ref, axis=1)
#     period_delta_states_ref = np.vstack((period_delta_positions_ref, period_delta_velocities_ref))
  
#     steady_state_periods_ref = 0
#     diffs_ref = np.diff(period_delta_states_ref, axis=1)
#     max_diffs_ref = np.mean(np.abs(diffs_ref), axis=0)

#     # Calculate the reference confidence between the two simulations (max absolute difference in delta state variables at final time)
#     confidence = np.max(np.abs(period_delta_states[:, -1] - period_delta_states_ref[:, -1]))

#     plt.figure()
#     plt.semilogy(max_diffs, 'o-')
#     # Horizontal line for confidence
#     plt.axhline(y=confidence, color='r', linestyle='--', label='Confidence Level')
#     plt.xlabel('Period Index')
#     plt.ylabel('Max Absolute Difference from Previous Period')
#     plt.title('Convergence to Periodic Steady State')
#     plt.grid(True)
#     # Save figure as PNG and PDF
#     plt.savefig('convergence_to_periodic_steady_state.pdf')
#     plt.savefig('convergence_to_periodic_steady_state.png')
#     # plt.show()


#     # plt.semilogy(max_diffs_ref, 'o-')
#     # plt.xlabel('Period Index')
#     # plt.ylabel('Max Absolute Difference from Previous Period')
#     # plt.title('Convergence to Periodic Steady State (Reference)')
#     # plt.grid(True)
#     # plt.show()



#     # Plot the results on subplots
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8)) 
#     # Plot positions of the cars over time relative to leading car
#     pos_lead = initial_positions[-1] + initial_velocities[-1] * t
#     for i in range(n):
#         axs[0].plot(t, x[2*i, :] - pos_lead, label=f'Position car {i+1} relative to unperturbed leading car')
#     axs[0].set_xlabel('Time (s)')
#     axs[0].set_ylabel('Relative Position (m)')
#     axs[0].set_title('Relative Position vs Time')      
#     # Plot velocities of the cars over time
#     for i in range(n):
#         axs[1].plot(t, x[2*i + 1, :], label=f'Velocity car {i+1}')
#     axs[1].set_xlabel('Time (s)')
#     axs[1].set_ylabel('Velocity (m/s)')
#     axs[1].set_title('Velocity vs Time')        
#     plt.tight_layout()


#     # Save the figure as a PNG file and a PDF file
#     fig.savefig('periodic_ss_simulation.png')
#     fig.savefig('periodic_ss_simulation.pdf')

#     plt.show()

if __name__ == "__main__":
    # Run shooting Newton to find periodic steady state
    # Generate parameters and initial conditions for n cars
    n = 100
    parameters = [Parameters(2.0, 0.5, 0.5, 0, 1.0, 3.0) for _ in range(n)]

    # Set the amplitude of the cosine input for the leading car
    parameters[-1].input_ampl = 3.0

    T = 5.0  # Period of the cosine input

    param_dict = {"parameters": parameters, "dxFD":1e-8, "T": T}
    # Set the positions initial conditions with nth car leading
    dx = 15.0
    initial_positions = [(i * dx) for i in range(n)]
    # Set all velocties to 10
    initial_velocities = [20.0 for _ in range(n)]
    

    initial_conditions = []
    for i in range(n):
        initial_conditions.append(initial_positions[i])
        initial_conditions.append(initial_velocities[i])

    dt = 0.05
    errf = 1e-6
    errDeltax = 1e-6
    relDeltax = 1e-6
    MaxIter = 10
    visualize = False
    FiniteDifference = False

    # Load the simulation data from a .npz file
    data = np.load('periodic_ss_simulation_20251129_190549.npz', allow_pickle=True)
    x = data['x']
    t = data['t']
    n = int(data['n'])
    dt = np.unique(np.round(np.diff(t), decimals=2)).item()
    # run eval_f_Shooting with initial conditions
    param_dict = {"parameters": parameters, "dxFD":1e-8, "T": T, "eval_f": eval_f, "eval_Jf": eval_Jf, "eval_u": gen_eval_u_cos(T), "dt": dt}
    F, X, t_sim = shooting_Newton.eval_f_Shooting(x[:, -1], param_dict)
    ic(np.max(np.abs(F)))
 
    time_init = datetime.now() 
    X_pss, t_pss, converged, errf_k, errDeltax_k, relDeltax_k, iterations = shooting_Newton.shooting_Newton(
        eval_f, x[:, -1], param_dict, gen_eval_u_cos(T), dt, errf, errDeltax, relDeltax, MaxIter, visualize, FiniteDifference, eval_Jf
    )
    time_final = datetime.now()

    total_duration = time_final - time_init
    ic(total_duration)

    # # Simulate the first period with X_pss as initial conditions to verify periodicity
    # x, t, k = implicit.implicit("Trapezoidal", eval_f, X_pss[:,0], param_dict, gen_eval_u_cos(T), 0.0, T, dt, False, True, use_GCR=True)

    # # Calculate the positions and velocities delta for each car at the end at each time step
    # delta_positions = np.zeros((n-1, x.shape[1]))
    # delta_velocities = np.zeros((n-1, x.shape[1]))
    
    # for i in range(n-1):
    #     delta_positions[i, :] = x[2*i + 2, :] - x[2*i, :]
    #     delta_velocities[i, :] = x[2*i + 3, :] - x[2*i + 1, :]

    # delta_states = np.vstack((delta_positions, delta_velocities))

    # # Save the simulation data to a .npz file with a timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"periodic_ss_shooting_simulation_{timestamp}.npz"
    # np.savez(filename, x=x, t=t, n=n, T=T, params=parameters[0].__dict__, input_ampl=parameters[-1].input_ampl, initial_positions=initial_positions, initial_velocities=initial_velocities, duration=total_duration.total_seconds())

    # # Verify convergence to periodic steady state by comparing initial and final states (delta_states)
    # state_difference = np.abs(delta_states[:, -1] - delta_states[:, 0])
    # ic(np.max(state_difference))

    # # Plot the results on subplots
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    # # Plot positions of the cars over time relative to leading car
    # pos_lead = initial_positions[-1] + initial_velocities[-1] * t
    # for i in range(n):
    #     axs[0].plot(t, x[2*i, :] - pos_lead, label=f'Position car {i+1} relative to unperturbed leading car')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Relative Position (m)')
    # axs[0].set_title('Relative Position vs Time')      
    # # Plot velocities of the cars over time
    # for i in range(n):
    #     axs[1].plot(t, x[2*i + 1, :], label=f'Velocity car {i+1}')
    # axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylabel('Velocity (m/s)')
    # axs[1].set_title('Velocity vs Time')        
    # plt.tight_layout()  
    # # Save the figure as a PNG file and a PDF file
    # fig.savefig('periodic_ss_shooting_simulation.png')
    # fig.savefig('periodic_ss_shooting_simulation.pdf')  
    # plt.show()



