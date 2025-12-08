import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_traffic(positions: np.ndarray, times: np.ndarray):
    """
    Animates a single traffic simulation given positions over time.
    """
    n_cars = positions.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, np.max(positions) + 10)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Position along road (m)')
    ax.set_yticks([])

    car_dots, = ax.plot([], [], 's', markersize=5)
    road_length = np.max(positions) + 10
    ax.plot([0, road_length], [0.2, 0.2], 'k-', linewidth=1)
    ax.plot([0, road_length], [-0.2, -0.2], 'k-', linewidth=1)

    def init():
        car_dots.set_data([], [])
        return car_dots

    def update(frame):
        x = positions[frame, :]
        y = np.zeros(n_cars)
        car_dots.set_data(x, y)
        return car_dots

    ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=100)
    plt.title('Traffic Simulation Animation')
    plt.show()