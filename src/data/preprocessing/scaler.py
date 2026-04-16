import numpy as np

class SpatialTemporalScaler:
    """
    Standardizes flow and pressure features across 782/388 nodes robustly.
    Since cyber-attacks introduce extreme deviations, we use median/MAD 
    (Median Absolute Deviation) instead of standard Mean/Std to prevent 
    adversarial outlier corruption influencing normalization statistics mathematically.
    """
    def __init__(self):
        self.median = None
        self.mad = None
        self.epsilon = 1e-8

    def fit(self, data: np.ndarray):
        """
        Calculates robust feature distributions.
        
        Args:
            data (np.ndarray): Tensor (T, N, F) representing attack-free training constraints.
        """
        # Collapse time and nodes to find global feature scales (F)
        feature_data = data.reshape(-1, data.shape[-1])
        
        self.median = np.median(feature_data, axis=0)
        self.mad = np.median(np.abs(feature_data - self.median), axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies mapping limits limiting deviations geometrically.
        """
        if self.median is None or self.mad is None:
            raise ValueError("Scaler must logically be fitted before mathematical transformation.")
            
        scaled_data = (data - self.median) / (self.mad + self.epsilon)
        return scaled_data
        
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Reverts geometries for precise physical loss evaluations (Eq 24 HW logic requires absolute units).
        """
        return (scaled_data * (self.mad + self.epsilon)) + self.median
