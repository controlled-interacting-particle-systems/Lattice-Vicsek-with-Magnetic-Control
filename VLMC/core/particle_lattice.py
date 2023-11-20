        density: float = 0.0,
        sinks: torch.Tensor = None,
        :param sinks: A binary matrix indicating the sink (absorption) locations.
        self.lattice = torch.zeros(
            (self.num_layers, height, width), dtype=torch.bool
        )  # the lattice is a 3D tensor with dimensions corresponding to layers/orientations, width, and height. orientation layers are up, down, left, and right in that order.
        if sinks is not None:
            self.add_layer(sinks, "sinks")

        # Initialize the lattice with particles at a given density.
        self.initialize_lattice(density)
