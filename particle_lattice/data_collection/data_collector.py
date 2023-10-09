import pandas as pd
import numpy as np
import copy

ORIENTATION_TO_VECTOR = {0: np.array([1, 0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1])}

class DataCollector:
    """
    Class for collecting and analyzing data from the simulation.
    
    Attributes:
        data (pd.DataFrame): A DataFrame to store collected data.
    """
    def __init__(self):
        # Initialize an empty DataFrame to store data
        self.data = pd.DataFrame(columns=['TimeStep', 'ElapsedTime', 'OrientationCounts', 'LatticeState'])

    def collect_data(self, simulation):
        """
        Collect data from the current state of the simulation.
        
        :param simulation: The simulation object.
        :type simulation: Simulation
        """
        # Collect data from simulation
        time_step = simulation.time_steps
        elapsed_time = simulation.t 
        orientation_counts = self._count_orientations(simulation.lattice)

        # Deep copy the lattice state so that future changes to the lattice don't affect stored data
        lattice_state = copy.deepcopy(simulation.lattice.grid)
        
        #validate that data is a dataframe
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        # Append to data DataFrame
        new_data = pd.DataFrame({'TimeStep': [time_step],
                                'ElapsedTime': [elapsed_time],
                                'OrientationCounts': [orientation_counts],
                                'LatticeState': lattice_state})

        self.data = pd.concat([self.data, new_data], ignore_index=True)
        
    def _count_orientations(self, lattice):
        """
        Count the number of particles with each orientation on the lattice.
        
        :param lattice: The lattice object.
        :type lattice: Lattice
        :return: Dictionary of orientation counts.
        :rtype: dict
        """
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for y in range(lattice.height):
            for x in range(lattice.width):
                particle = lattice.grid[y, x]
                if particle is not None:
                    orientation = particle.get_orientation()
                    counts[orientation] += 1
        return counts
    
    def _orientation_vector_from_counts(self, counts):
        """
        Compute the average orientation vector from the orientation counts.
        
        :param counts: Dictionary of orientation counts.
        :type counts: dict
        :return: Average orientation vector.
        :rtype: np.ndarray
        """
        # Compute the average orientation vector
        orientation_vectors = [ORIENTATION_TO_VECTOR[orientation] * count for orientation, count in counts.items()]
        average_orientation_vector = sum(orientation_vectors) / sum(counts.values())
        return average_orientation_vector
    

    def analyze_data(self):
        """
        Perform post-processing analysis on the collected data.
        """
        # compute the order parameter: module of the average orientation vector

        # compute the average orientation vector for each time step
        orientation_vectors = self.data['OrientationCounts'].apply(self._orientation_vector_from_counts)
        order_parameters = orientation_vectors.apply(np.linalg.norm) 
        self.data['OrderParameter'] = order_parameters

        # compute the cumulative mean order parameter for each time step
        self.data['CumulativeMeanOrderParameter'] = self.data['OrderParameter'].expanding().mean()

        # plot the cumulative mean order parameter as a function of time
        self.data.plot(x='ElapsedTime', y='CumulativeMeanOrderParameter', legend=None)


