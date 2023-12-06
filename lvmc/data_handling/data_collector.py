import pandas as pd
import numpy as np
import copy

ORIENTATION_TO_VECTOR = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}


class DataCollector:
    """
    Class for collecting and analyzing data from the simulation.

    Attributes:
        event_data (pd.DataFrame): A DataFrame to store event data. Each row represents an event, and the columns are "TimeStep", "dt", "EventType", "X", "Y", and magnetic field direction.
        metadata (dict): A dictionary to store metadata.
        initial_state (np.ndarray): The initial state of the lattice.
    """

    def __init__(self):
        # Initialize an empty DataFrame to store data
        self.event_data = pd.DataFrame(
            columns=["TimeStep", "dt", "EventType", "X", "Y", "MagneticField"]
        )
        self.initial_state = None
        self.metadata = {}

    def collect_data(self, simulation):
        """
        Collect data from the current state of the simulation.

        :param simulation: The simulation object.
        :type simulation: Simulation
        """
        # Collect data from simulation
        self.initial_state = copy.deepcopy(np.array(simulation.lattice.lattice))
        elapsed_time = simulation.t
