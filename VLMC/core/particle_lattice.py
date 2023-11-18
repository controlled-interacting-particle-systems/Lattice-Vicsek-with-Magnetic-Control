        TM_up = self.lattice[self.layer_indices["up"]] * empty_cells.roll(shifts=1, dims=0)
        TM_down = self.lattice[self.layer_indices["down"]] * empty_cells.roll(shifts=-1, dims=0)
        TM_left = self.lattice[self.layer_indices["left"]] * empty_cells.roll(shifts=1, dims=1)
        TM_right = self.lattice[self.layer_indices["right"]] * empty_cells.roll(shifts=-1, dims=1)
