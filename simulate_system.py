import matplotlib.pyplot as plt

from provided_solvers.SimpleSolver import SimpleSolver
from evaluate_f import eval_f, Parameters

if __name__ == "__main__":
    parameters = [Parameters(1.0, 1.0, 1.0), Parameters(1.0, 1.0, 1.5), Parameters(1.0, 1.0, 2.0), Parameters(1.0, 1.0, 1.0)]
    x, t = SimpleSolver(eval_f, [0.0, 1.0, 10.0, 1.0, 15.0, 0.5, 15.2, 0.5], parameters, lambda t: None, 1000, 0.01, visualize=False)
    
    # Plot the results on subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot positions of the cars over time
    axs[0].plot(t, x[0, :], label='Position car 1')
    axs[0].plot(t, x[2, :], label='Position car 2')
    axs[0].plot(t, x[4, :], label='Position car 3')
    axs[0].plot(t, x[6, :], label='Position car 4')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('Position vs Time')
    axs[0].legend()

    # Plot velocities of the cars over time
    axs[1].plot(t, x[1, :], label='Velocity car 1')
    axs[1].plot(t, x[3, :], label='Velocity car 2')
    axs[1].plot(t, x[5, :], label='Velocity car 3')
    axs[1].plot(t, x[7, :], label='Velocity car 4')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('Velocity vs Time')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Print final distances between cars
    for i in range(3):
        distance = x[2*i+2, -1] - x[2*i, -1]
        print(f"Final distance between car {i+1} and car {i+2}: {distance:.4f} m")
        print(f"Expected final distance between car {i+1} and car {i+2}: {x[2*i+1, -1]*parameters[i].tau:.4f} m")
        print(f"Final velocity of car {i+1}: {x[2*i+1, -1]:.4f} m/s")
        print()
