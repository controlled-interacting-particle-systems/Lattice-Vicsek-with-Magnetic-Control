# Particle Lattice Simulation

## Model Overview

The core of this simulation is based on a model where particles move on a two-dimensional lattice with periodic boundary conditions. This model is inspired by the research paper ["Traffic jams, gliders, and bands in the quest for collective motion"](https://arxiv.org/pdf/1302.3797.pdf) by Fernando Peruani, Tobias Klauss, Andreas Deutsch, and Anja Voss-Boehme.

Each particle has one of four possible orientations: up, down, left, or right. The orientation of a particle fully determines its moving direction. Particles have two fundamental actions:

1. **Reorientation**: Change their orientation based on the orientations of their nearest neighbors.
2. **Migration**: Move to an adjacent lattice cell in the direction of their orientation.

The particles' actions are dictated by transition rates (`TR` for reorientation, `TM` for migration). The simulation includes the effect of an external magnetic field that can rotate the particles.

### Control Mechanism in the Lattice Vicsek Model

![Control Mechanism and Tasks in Lattice Vicsek Model](https://github.com/zakaryael/MagneticVicsekLattice/blob/main/control_mechanism_and_tasks.png)

This visual aid showcases the control mechanism introduced in the Lattice Vicsek Model. A simple magnetic field effect is used for 90-degree rotations. The tasks involve directing particles to either move towards or avoid specific target routes.


### Citation

For the underlying model and theoretical background, see:

- **Traffic jams, gliders, and bands in the quest for collective motion**  
  Fernando Peruani, Tobias Klauss, Andreas Deutsch, Anja Voss-Boehme  
  [ArXiv Paper](https://arxiv.org/pdf/1302.3797.pdf)


### Simulation Example

![vicsek with no control](https://github.com/zakaryael/MagneticVicsekLattice/blob/main/example_animation.gif)


<figure markdown>
<figcaption>Lattice Vicsek particles without control on a simple periodic rectangular lattice</figcaption>
</figure>


## Installation

Clone the repository and navigate into the project directory. 
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Install the 'particle_lattice' package using pip:

```bash
pip install -e .
```

## Usage

Example usage scripts are provided in the `examples/` directory.

Basic usage:

tbd

## Modules

### Core

- `particle_lattice.py` 
- `simulation.py`: Handles the main simulation loop
- `magnetic_field.py`: Manages magnetic field effects on particles

#### TODO
- [ ] store rates in the simulation class to improve effeciency
- [ ] add a list of particles attribute to the lattice class to improve effeciency
- [ ] add an apply method to the magnetic field class for increased modularity
- [ ] implement different boundary conditions

#### TODO
- [  ] post-simulation visualizations
- [  ] add visualization for magnetic field

### Examples

- `examples/`: Example usage scripts

### Tests

#### TODO
- [ ] add unit tests for core modules
  - [x] unit tests for particle_lattice.py
  - [ ] unit tests for simulation.py
  - [ ] unit tests for magnetic_field.py


## Future Extensions

