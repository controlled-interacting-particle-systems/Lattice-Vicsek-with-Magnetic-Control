        TM_up = self.lattice[self.layer_indices["up"]] * empty_cells.roll(shifts=1, dims=0)
        TM_down = self.lattice[self.layer_indices["down"]] * empty_cells.roll(shifts=-1, dims=0)
        TM_left = self.lattice[self.layer_indices["left"]] * empty_cells.roll(shifts=1, dims=1)
        TM_right = self.lattice[self.layer_indices["right"]] * empty_cells.roll(shifts=-1, dims=1)
        # Convolve each orientation layer of the lattice using layer indices
        for orientation, index in self.layer_indices.items():
            input_tensor = self.lattice[index].unsqueeze(0).unsqueeze(0).float()
            TR_tensor[index] = F.conv2d(input_tensor, kernel, padding=1)[0, 0]
        up_index, down_index = self.layer_indices["up"], self.layer_indices["down"]
        left_index, right_index = self.layer_indices["left"], self.layer_indices["right"]

        TR_tensor[up_index], TR_tensor[down_index] = (
            TR_tensor[up_index] - TR_tensor[down_index],
            TR_tensor[down_index] - TR_tensor[up_index],
        TR_tensor[left_index], TR_tensor[right_index] = (
            TR_tensor[left_index] - TR_tensor[right_index],
            TR_tensor[right_index] - TR_tensor[left_index],
