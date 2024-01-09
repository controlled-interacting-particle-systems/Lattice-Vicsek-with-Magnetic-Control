import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Visualization:
    def __init__(self, data_source=None):
        self.data_source = data_source
        self.ax = None  # Placeholder for the axes object
        if data_source:
            self.setup_static_elements()

    def setup_static_elements(self):
        """
        Set up static elements in the plot, such as obstacles and sinks.
        """
        fig, self.ax = plt.subplots()
        obstacles = self.data_source.data["obstacles"]
        sinks = self.data_source.data["sinks"]

        for y in range(obstacles.shape[0]):
            for x in range(obstacles.shape[1]):
                if obstacles[y, x]:
                    self.ax.add_patch(
                        patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="black")
                    )
                if sinks[y, x]:
                    self.ax.add_patch(
                        patches.Circle(
                            (x, y), radius=0.3, facecolor="white", edgecolor="black"
                        )
                    )

    def draw_orientation(self, x, y, orientation):
        size = 0.4  # Size of the triangle
        orientations = [
            (np.pi, "blue"),
            (-np.pi / 2, "green"),
            (0, "red"),
            (np.pi / 2, "yellow"),
        ]
        if orientation < 4:
            angle, color = orientations[orientation]
            return patches.RegularPolygon(
                (x, y),
                numVertices=3,
                radius=size,
                orientation=angle,
                edgecolor="none",
                facecolor=color,
            )

    def update_plot(self, snapshot):
        """
        Update the plot with the current snapshot of the simulation.

        :param snapshot: The current state of the simulation.
        """
        if not self.ax:
            raise ValueError("Plot axes not initialized.")

        self.ax.clear()

        for y in range(snapshot.shape[1]):
            for x in range(snapshot.shape[2]):
                for orientation in range(4):
                    if snapshot[orientation, y, x]:
                        triangle = self.draw_orientation(x, y, orientation)
                        self.ax.add_patch(triangle)

        self.ax.set_xlim(0, snapshot.shape[2])
        self.ax.set_ylim(0, snapshot.shape[1])
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()

    def real_time_visualization(self, interval=50):
        if self.data_source is None:
            raise ValueError("Data source is not provided for real-time visualization.")

        fig, ax = plt.subplots()
        plt.ion()  # Turn on interactive mode

        for frame in range(len(self.data_source.data["snapshots"])):
            snapshot = self.data_source.data["snapshots"][frame][1]
            self.update_plot(snapshot, ax)
            plt.draw()
            plt.pause(interval / 1000.0)  # interval is in milliseconds

        plt.ioff()  # Turn off interactive mode

    def post_processing_visualization(self, hdf5_file, interval=50):
        with h5py.File(hdf5_file, "r") as file:
            fig, ax = plt.subplots()
            snapshots = file["snapshots"]

            for i in range(len(snapshots.keys())):
                print(
                    f"Processing snapshot {i + 1}/{len(snapshots.keys())}"
                )  # Debug print
                snapshot = snapshots[f"snapshot_{i}"][()]
                self.update_plot(snapshot)
                plt.draw()
                plt.pause(interval / 1000.0)  # interval is in milliseconds
            plt.show()
