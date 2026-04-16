import numpy as np

class DiurnalDemandGenerator:
    """
    Extracts and maps exogenous base demands and diurnal profiles.
    Used mathematically as \hat{D}_{j,t} = D_j^{base} \times P_m(t).
    Critically avoids cyclical dependency in Mass Balance physical logic evaluations.
    """
    def __init__(self, base_demands: np.ndarray, historical_profiles: np.ndarray):
        """
        Args:
            base_demands (np.ndarray): Historical nodal baselines D_j^{base} (shape: num_nodes,)
            historical_profiles (np.ndarray): P_m(t) multipliers representing hour-specific shifts
        """
        self.base_demands = base_demands
        self.profiles = historical_profiles

    def get_demand(self, node_idx: int, time_step: int, day_type: int) -> float:
        """
        Calculates the expected purely exogenous demand for a node at a given time.
        
        Args:
            node_idx (int): Global index j for the target node.
            time_step (int): Current temporal evaluation period t.
            day_type (int): Profile type m (e.g. 0=Weekday, 1=Weekend).
            
        Returns:
            float: Expected demand \hat{D}_{j,t} in standard physical units.
        """
        # Obtain specific time-of-day profile scaling multiplier
        time_index = time_step % self.profiles.shape[1] 
        multiplier = self.profiles[day_type, time_index]
        
        return self.base_demands[node_idx] * multiplier

    def generate_full_matrix(self, total_steps: int, day_types_sequence: np.ndarray) -> np.ndarray:
        """
        Broadcasting generation for the full prediction matrix.
        
        Args:
            total_steps (int): Total prediction length.
            day_types_sequence (np.ndarray): Sequence mapping intervals to day types.
            
        Returns:
            np.ndarray: Matrix of demands shape (total_steps, num_nodes)
        """
        num_nodes = len(self.base_demands)
        demand_matrix = np.zeros((total_steps, num_nodes))
        
        for t in range(total_steps):
            day_type = day_types_sequence[t]
            time_index = t % self.profiles.shape[1]
            multiplier = self.profiles[day_type, time_index]
            demand_matrix[t, :] = self.base_demands * multiplier
            
        return demand_matrix
