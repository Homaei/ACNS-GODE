import numpy as np
import torch

class PacketLossImputer:
    """
    Simulates SCADA Zero-Order Hold (ZOH) interpolation. 
    Strictly necessary for managing discrete observation bounds during missing packet deliveries.
    This maintains the integrity of the A(tau) step function within continuous ODE integrals.
    """
    @staticmethod
    def apply_zoh(data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Holds physical states mathematically constant until next sensor packet arrives.
        
        Args:
            data (np.ndarray): The raw array potentially filled with NaNs or zeros.
            missing_mask (np.ndarray): Boolean mask indicating packet intervals.
            
        Returns:
            np.ndarray: Topologically preserved continuous boundaries constraints.
        """
        imputed = np.copy(data)
        for t in range(1, imputed.shape[0]):
            for n in range(imputed.shape[1]):
                if missing_mask[t, n]:
                    # ZOH limits simply preserve the previous bounds
                    imputed[t, n] = imputed[t-1, n]
        return imputed
