from lvmc.data_handling.visualization import Visualization


def run_post_processing_viz(hdf5_file):
    visualizer = Visualization()
    visualizer.post_processing_visualization(hdf5_file)


if __name__ == "__main__":
    hdf5_file_path = "simulation_data.hdf5"  # Replace with your HDF5 file path
    run_post_processing_viz(hdf5_file_path)
