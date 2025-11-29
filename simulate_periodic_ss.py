import matplotlib.pyplot as plt

from provided_solvers import forward_euler, implicit
from provided_solvers.eval_u_cos import gen_eval_u_cos

from evaluate_f import eval_f, Parameters

from icecream import ic

if __name__ == "__main__":
    # Generate parameters and initial conditions for n cars
    n = 20
    parameters = [Parameters(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) for _ in range(n)]

    # Set the amplitude of the cosine input for the leading car
    parameters[-1].input_ampl = 3.0

    T = 10.0  # Period of the cosine input

    param_dict = {"parameters": parameters, "dxFD":1e-8}
    # Set the positions initial conditions with nth car leading
    dx = 15.0
    initial_positions = [(i * dx) for i in range(n)]
    # Set all velocties to 10
    initial_velocities = [10.0 for _ in range(n)]
    

    initial_conditions = []
    for i in range(n):
        initial_conditions.append(initial_positions[i])
        initial_conditions.append(initial_velocities[i])

    t_start = 0.0
    t_end = 50.0
    # Simulate the system using Forward Euler method
    # x, t = forward_euler.forward_euler(eval_f, initial_conditions, param_dict, gen_eval_u_cos(T), t_start, t_end, 0.0001, visualize=False)  

    # Simulate the system using Implicit method backward Euler
    x, t, k = implicit.implicit("Trapezoidal", eval_f, initial_conditions, param_dict, gen_eval_u_cos(T), t_start, t_end, 0.05, False, True, use_GCR=True)

    # Plot the results on subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot positions of the cars over time relative to leading car
    for i in range(n):
        # axs[0].plot(t, x[2*i, :], label=f'Position car {i+1}')
        axs[0].plot(t, x[2*i, :] - x[2*(n-1), :], label=f'Position car {i+1} relative to car {n}')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('Position vs Time')
    # axs[0].legend()

    # Plot velocities of the cars over time
    for i in range(n):
        axs[1].plot(t, x[2*i + 1, :], label=f'Velocity car {i+1}')
        # axs[1].plot(t, x[2*i + 1, :] - x[2*(n-1) + 1, :], label=f'Velocity car {i+1} relative to car {n}')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('Velocity vs Time')
    # axs[1].legend()

    plt.tight_layout()
    plt.show()