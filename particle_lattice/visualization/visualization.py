import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


class Visualization:
    """
    Class to manage the animated visualization of the 2D lattice and particle orientations.

    Attributes:
        lattice (Lattice): The lattice object.
    """

    def __init__(self, lattice):
        self.lattice = lattice

    def draw_triangle(self, x, y, orientation, color, ax):
        """Draw a custom triangle based on orientation."""
        size = 0.2
        if orientation == 0:  # Up
            vertices = [(x, y + size), (x - size, y - size), (x + size, y - size)]
        elif orientation == 1:  # Down
            vertices = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        elif orientation == 2:  # Left
            vertices = [(x - size, y), (x + size, y - size), (x + size, y + size)]
        elif orientation == 3:  # Right
            vertices = [(x + size, y), (x - size, y - size), (x - size, y + size)]

        polygon = Polygon(vertices, closed=True, facecolor="none", edgecolor=color)
        ax.add_patch(polygon)

    def animate_lattice(self, num_frames, time_interval, simulation):
        """
        Animate the lattice with particles and their orientations.

        :param num_frames: Number of frames for the animation.
        :type num_frames: int
        :param time_interval: Time interval between frames.
        :type time_interval: int
        :param simulation: Simulation object to update particle states.
        :type simulation: Simulation
        """
        fig, ax = plt.subplots()
        orientation_colors = [
            "r",
            "g",
            "b",
            "m",
        ]  # Different colors for each orientation

        def update(frame):
            ax.clear()
            simulation.run_time_step()

            for y in range(self.lattice.height):
                for x in range(self.lattice.width):
                    particle = self.lattice.grid[y, x]
                    if particle is not None:
                        orientation = particle.get_orientation()
                        color = orientation_colors[orientation]
                        self.draw_triangle(x, y, orientation, color, ax)

            ax.set_aspect("equal", "box")
            ax.set_xlim(-1, self.lattice.width)
            ax.set_ylim(-1, self.lattice.height)

            # Display the elapsed time
            elapsed_time_str = f"Elapsed Time: {simulation.t:.2f}"
            time_steps_str = f"Time Steps: {simulation.time_steps}"
            ax.set_title(elapsed_time_str + "\n" + time_steps_str)

        ani = FuncAnimation(fig, update, frames=num_frames, interval=time_interval)
        plt.show()
