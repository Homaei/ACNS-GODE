import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
import numpy as np

class WDNTimeSeriesDataset(Dataset):
    """
    Base abstraction for Water Distribution Network Time-Series datasets.
    Provides standard capabilities for handling SCADA intervals, missing data mapping,
    and sequence extraction for graph Neural ODE models.
    """
    def __init__(self, 
                 sensor_readings: np.ndarray, 
                 schedules: np.ndarray, 
                 demands: np.ndarray, 
                 window_size: int = 2016, # Default to 1-week of 5-min intervals
                 is_train: bool = True):
        """
        Initialization for spatial-temporal WDN datasets.
        
        Args:
            sensor_readings (np.ndarray): Tensor of shape (time_steps, num_nodes, num_features)
            schedules (np.ndarray): Operational schedules for pumps/valves shape (time_steps, num_edges)
            demands (np.ndarray): Exogenous demand vectors shape (time_steps, num_nodes)
            window_size (int): Temporal window length for model historical context.
            is_train (bool): Determines if standard scaling statistics are extracted or applied.
        """
        super().__init__()
        self.sensor_readings = torch.FloatTensor(sensor_readings)
        self.schedules = torch.FloatTensor(schedules)
        self.demands = torch.FloatTensor(demands)
        self.window_size = window_size
        self.is_train = is_train

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        """
        Ensures consistent temporal dimensions across sensory data, schedules, and demands.
        Throws a ValueError if unaligned.
        """
        assert len(self.sensor_readings) == len(self.schedules) == len(self.demands), \
            "Temporal inconsistency: Sensor readings, schedules, and demands must share identical time dimensions."

    def __len__(self) -> int:
        """
        Returns total viable sequential batches.
        """
        return len(self.sensor_readings) - self.window_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Extracts a temporal sequence suitable for training the continuous-time latent dynamics.
        
        Returns:
            Dict containing:
            - 'x': Sensor histories (window_size, num_nodes, num_features)
            - 'schedule': Topology override binary vectors (window_size, num_edges)
            - 'demand': Exogenous baseline profile constraints (window_size, num_nodes)
            - 'y': The target ground truth vector for state t + 1 (num_nodes, num_features)
        """
        end_idx = idx + self.window_size
        
        # Sequence slicing
        sequence_x = self.sensor_readings[idx:end_idx]
        sequence_schedule = self.schedules[idx:end_idx]
        sequence_demand = self.demands[idx:end_idx]
        
        # Target is the immediately following step (t + 1)
        target_y = self.sensor_readings[end_idx]

        return {
            'x': sequence_x,
            'schedule': sequence_schedule,
            'demand': sequence_demand,
            'y': target_y
        }
