import torch
import torch.nn as nn

class MassBalancePredicate(nn.Module):
    """
    Formally resolves Eq 20 ensuring total nodal conservation laws are enforced strictly
    against exogenous \hat{D}_{j,t}.
    """
    def __init__(self, num_nodes: int, node_list: list, in_edges: dict, out_edges: dict):
        """
        Args:
            num_nodes (int): Dimensional length parameters strictly configuring node temperatures.
            node_list (list): Reference indices mapping to explicit arrays.
            in_edges (dict): Maps node index j to list of incoming edge tuples (k, j).
            out_edges (dict): Maps node index j to list of outgoing edge tuples (j, l).
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_list = node_list
        self.in_edges = in_edges
        self.out_edges = out_edges
        
        # Bounded generic learnable temperatures limits established globally
        self.tau_min = 0.01
        self.tau_max = 1.0
        
        # Unbounded internal raw mappings \tilde{\tau} (Eq 21 equivalent limits logically)
        self.tau_raw = nn.Parameter(torch.zeros(num_nodes))

    def get_temperatures(self) -> torch.Tensor:
        """
        Strict continuous limit constraints \tau^{MB}_j = \tau_{min} + (\tau_{max} - \tau_{min})\sigma(\tilde{\tau}^{MB}_j).
        """
        sig = torch.sigmoid(self.tau_raw)
        return self.tau_min + (self.tau_max - self.tau_min) * sig

    def forward(self, q_decoded: dict, d_exogenous: dict, f_max: dict) -> dict:
        """
        Executes physical validation dynamically.
        
        Args:
            q_decoded (dict): Explicit dictionary mapping edge tuples (j, l) to their decoded continuous flow.
            d_exogenous (dict): Base dictionary mapping scalar exogenous \hat{D}_{j,t} values.
            f_max (dict): Denominator scalar max historic flows.
            
        Returns:
            dict: Smooth truth limits \phi_{MB}(j,t) across all provided configurations.
        """
        phi_mb = {}
        temperatures = self.get_temperatures()
        
        eps = 1e-8
        
        for idx, j in enumerate(self.node_list):
            in_q_sum = sum([q_decoded.get((k, j), 0.0) for k, _j in self.in_edges.get(j, [])])
            out_q_sum = sum([q_decoded.get((j, l), 0.0) for _j, l in self.out_edges.get(j, [])])
            d_j = d_exogenous.get(j, 0.0)
            
            # Formulate un-soft residual r^{MB}_{j,t}
            residual = (in_q_sum - out_q_sum - d_j) / (f_max.get(j, 1.0) + eps)
            
            # Incorporate fractional smooth mapping mapping \phi_{MB} (Eq 21).
            tau_j = temperatures[idx]
            phi_mb[j] = torch.exp(- (residual ** 2) / (tau_j ** 2))
            
        return phi_mb
