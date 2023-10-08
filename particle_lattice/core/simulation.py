import numpy as np
import random
import math
from particle_lattice.core.particle import Particle
from particle_lattice.core.lattice import Lattice
from particle_lattice.core.magnetic_field import MagneticField


class Simulation:
    """
    Class to manage and run the simulation of particles on a 2D lattice using the Gillespie algorithm.
    
    Attributes:
        width (int): Width of the lattice.
        height (int): Height of the lattice.
        lattice (Lattice): Lattice object representing the grid.
        particles (list): List of Particle objects.
        v0 (float): Base rate for migration.
        g (float): Parameter controlling alignment sensitivity.
    """

    def __init__(self, width: int, height: int, v0: float, g: float, density: float, magnetic_field_interval: float=1):
        """
        Initialize the Simulation class.
        
        :param width: Width of the lattice.
        :type width: int
        :param height: Height of the lattice.
        :type height: int
        :param v0: Base rate for migration.
        :type v0: float
        :param g: Parameter controlling alignment sensitivity.
        :type g: float
        :param density: Density of particles in the lattice.
        :type density: float
        """
        self.width = width
        self.height = height
        self.lattice = Lattice(width, height)
        self.particles = []
        self.v0 = v0
        self.g = g
        self.density = density
        self.t = 0.0
        self.time_steps = 0
        self._initialize_particles()
        self.magnetic_field = MagneticField()
        self.magnetic_field_interval = magnetic_field_interval
        self.next_magnetic_field_application = magnetic_field_interval
    
    def _initialize_particles(self):
        """
        Randomly initialize particles on the lattice based on the given density.
        """
        total_nodes = self.width * self.height
        num_particles = int(total_nodes * self.density)

        for _ in range(num_particles):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

            # Ensure the node is empty
            while not self.lattice.is_node_empty(x, y):
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

            orientation_code = random.randint(0, 3)
            particle = Particle(x, y, orientation_code)
            self.particles.append(particle)
            self.lattice.place_particle(particle, x, y)



    def next_event_time(self) -> float:
        """
        Calculate the time for the next event using exponential distribution.
        
        :return: Time for the next event.
        :rtype: float
        """
        rate_sum = sum(p.compute_TR(self.lattice, new_orientation, self.g) +
                       p.compute_TM(self.lattice, self.v0) for p in self.particles
                       for new_orientation in range(4))
        
        return -math.log(random.random()) / rate_sum

    def choose_event(self) -> tuple:
        """
        Choose the next event (either reorientation or migration) based on the transition rates.
        
        :return: Particle and action ('reorient' or 'migrate') as a tuple.
        :rtype: tuple
        """
        rates = []
        events = []

        # Calculate all possible transition rates and associated events
        for particle in self.particles:
            for new_orientation in range(4):
                rate = particle.compute_TR(self.lattice, new_orientation, self.g)
                rates.append(rate)
                events.append((particle, 'reorient', new_orientation))

            rate = particle.compute_TM(self.lattice, self.v0)
            rates.append(rate)
            events.append((particle, 'migrate'))

        # Randomly choose an event based on the transition rates
        total_rate = sum(rates)
        rand_threshold = random.random() * total_rate
        accumulated_rate = 0.0

        for rate, event in zip(rates, events):
            accumulated_rate += rate
            if accumulated_rate >= rand_threshold:
                return event


    def run_time_step(self):
        """
        Run a single time step of the simulation.
        """
        dt = self.next_event_time()
        particle, action, *args = self.choose_event()
        
        if action == 'reorient':
            new_orientation = args[0]
            particle.reorient(new_orientation)
        elif action == 'migrate':
            # Calculate new position based on the particle's orientation
            dx, dy = particle.get_orientation_vector()
            new_x, new_y = (particle.x + dx) % self.width, (particle.y + dy) % self.height

            # Move the particle to the new position
            self.lattice.place_particle(None, particle.x, particle.y)
            particle.move(new_x, new_y)
            self.lattice.place_particle(particle, new_x, new_y)
        
        # Update the next magnetic field application time
        self.next_magnetic_field_application -= self.t
        
        # Apply magnetic field if it's time
        if self.next_magnetic_field_application <= 0:
            for y in range(self.lattice.height):
                for x in range(self.lattice.width):
                    particle = self.lattice.grid[y, x]
                    if particle is not None:
                        self.magnetic_field.rotate_particle(particle)
            self.next_magnetic_field_application += self.magnetic_field_interval

        
        # Update the elapsed time and time steps
        self.t += dt    
        self.time_steps += 1