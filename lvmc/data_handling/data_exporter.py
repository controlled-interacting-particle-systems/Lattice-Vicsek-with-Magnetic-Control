import h5py
import json
import torch
from lvmc.data_handling.data_collector import DataCollector


class DataExporter:
    def __init__(self, filename: str, data_collector: "DataCollector") -> None:
        """
        Initialize an exporter for saving simulation data in HDF5 format.

        :param filename: The name of the file where data will be saved.
        :param data_collector: The data collector object containing simulation data.
        """
        self.filename = filename
        self.data_collector = data_collector

    def export_data(self) -> None:
        """
        Export the collected data to an HDF5 file.

        This method serializes and stores metadata, initial configuration,
        snapshots of the simulation, and event data in the specified HDF5 file.
        """
        with h5py.File(self.filename, "w") as file:
            # Serialize and store metadata
            for key, value in self.data_collector.data["metadata"].items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                file.attrs[key] = value

            # Export initial configuration
            if isinstance(self.data_collector.data["initial_config"], torch.Tensor):
                file.create_dataset(
                    "initial_config",
                    data=self.data_collector.data["initial_config"].numpy(),
                )
            else:
                file.create_dataset(
                    "initial_config", data=self.data_collector.data["initial_config"]
                )

            # Export snapshots
            snapshots_group = file.create_group("snapshots")
            for i, (time, snapshot) in enumerate(self.data_collector.data["snapshots"]):
                if isinstance(snapshot, torch.Tensor):
                    snapshot = snapshot.numpy()
                snapshots_group.create_dataset(f"snapshot_{i}", data=snapshot)
                snapshots_group[f"snapshot_{i}"].attrs["time"] = time

            # Export events
            events_group = file.create_group("events")
            for i, event in enumerate(self.data_collector.data["events"]):
                event_dataset = events_group.create_dataset(
                    f"event_{i}", data=list(event.values())
                )
                for j, key in enumerate(event.keys()):
                    event_dataset.attrs[key] = event[key]
