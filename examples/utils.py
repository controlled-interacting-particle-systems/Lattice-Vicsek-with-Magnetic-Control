import torch

def generate_binary_matrix(height, width, topology_name, walls=None):
    """
    Generate a binary matrix representing a specified lattice topology.

    Parameters
    ----------
    height : int
        The height of the lattice.
    width : int
        The width of the lattice.
    topology_name : str
        The name of the topology to generate. Options include "checkerboard",
        "horizontal_stripes", "vertical_stripes", "central", "corners", and "random".
    walls : list of str, optional
        A list specifying which walls to add. Options in the list can include
        "top", "bottom", "left", and "right". Default is None, meaning no walls.

    Returns
    -------
    torch.Tensor
        A 2D Torch tensor of shape (height, width) representing the lattice topology,
        where 1 indicates the presence of an obstacle or a sink, and 0 indicates an empty cell.

    Examples
    --------
    >>> generate_binary_matrix(10, 10, "checkerboard")
    tensor([[1, 0, 1, 0, ...],
            [0, 1, 0, 1, ...],
            ...])
    """
    matrix = torch.zeros((height, width), dtype=torch.bool)

    # Define the topologies
    if topology_name == "checkerboard":
        matrix[::2, ::2] = 1
        matrix[1::2, 1::2] = 1

    elif topology_name == "horizontal_stripes":
        matrix[::2, :] = 1

    elif topology_name == "vertical_stripes":
        matrix[:, ::2] = 1

    elif topology_name == "central":
        center_x, center_y = width // 2, height // 2
        matrix[center_y, center_x] = 1

    elif topology_name == "corners":
        matrix[0, 0] = matrix[0, -1] = matrix[-1, 0] = matrix[-1, -1] = 1

    elif topology_name == "random":
        matrix = torch.randint(0, 2, (height, width), dtype=torch.bool)

    # Add walls if specified
    if walls:
        if "top" in walls:
            matrix[0, :] = 1
        if "bottom" in walls:
            matrix[-1, :] = 1
        if "left" in walls:
            matrix[:, 0] = 1
        if "right" in walls:
            matrix[:, -1] = 1

    return matrix

def generate_lattice_topology(height, width, obstacle_topology, sink_topology, obstacle_walls=None, sink_walls=None):
    """
    Generate lattice topologies for obstacles and sinks with optional walls.

    Parameters
    ----------
    height : int
        The height of the lattice.
    width : int
        The width of the lattice.
    obstacle_topology : str
        The topology name for obstacles.
    sink_topology : str
        The topology name for sinks.
    obstacle_walls : list of str, optional
        Walls to add to the obstacle topology. Options are "top", "bottom", "left", "right".
    sink_walls : list of str, optional
        Walls to add to the sink topology. Options are "top", "bottom", "left", "right".

    Returns
    -------
    tuple of torch.Tensor
        A tuple containing two 2D Torch tensors. The first tensor represents the obstacle
        topology, and the second represents the sink topology.

    Examples
    --------
    >>> obstacles, sinks = generate_lattice_topology(10, 10, "checkerboard", "central", ["top", "left"], ["bottom", "right"])
    >>> obstacles
    tensor([[1, 1, 1, 1, ...],
            [1, 0, 0, 0, ...],
            ...])
    >>> sinks
    tensor([[0, 0, 0, 0, ...],
            [0, 0, 0, 0, ...],
            ...])
    """
    obstacles = generate_binary_matrix(height, width, obstacle_topology, obstacle_walls)
    sinks = generate_binary_matrix(height, width, sink_topology, sink_walls)

    # Ensure sinks do not overlap with obstacles
    sinks = torch.logical_and(sinks, torch.logical_not(obstacles))

    return obstacles, sinks

