import torch
import torch.nn as nn

class PhysicalFusion(nn.Module):
    """
    Executes geometric constraint logic mapping (Eq 26).
    Enforces simultaneous verification ensuring nodes satisfy identical topological parameters concurrently.
    """
    def __init__(self):
        super().__init__()

    def forward(self, node_list: list, in_edges: dict, out_edges: dict, 
                phi_mb: dict, phi_hw: dict, phi_pump: dict) -> dict:
        """
        Derives the fundamental \alpha_{j,t} non-conformity limit mapping across targets.
        
        Args:
            node_list (list): Base sequence establishing calculation scope.
            in_edges (dict): Reversal lookup tuples specifying inbound relations.
            out_edges (dict): Standard lookup defining outbound dependencies.
            phi_mb (dict): Solved bounds directly mapping node geometries.
            phi_hw (dict): Extensively computed limits across pipe boundaries.
            phi_pump (dict): Special manufacturer limitations explicitly defined.
            
        Returns:
            dict: \alpha_{j,t} limiting structures constrained explicitly between [0, 1].
        """
        alpha_scores = {}
        
        for j in node_list:
            # Gather local edges connected logically
            incident = in_edges.get(j, []) + out_edges.get(j, [])
            edge_scores = []
            
            # Collect valid bounding constraints \phi_{HW} and \phi_{pump} explicitly.
            for edge in incident:
                if edge in phi_hw:
                    edge_scores.append(phi_hw[edge])
                if edge in phi_pump:
                    edge_scores.append(phi_pump[edge])
                    
            if len(edge_scores) > 0:
                bar_phi_hw = torch.mean(torch.stack(edge_scores))
            else:
                bar_phi_hw = torch.tensor(1.0) # Default to perfect conformity if no active edges exist
                
            phi_mb_val = phi_mb.get(j, torch.tensor(1.0))
            
            # Geometric mean ensures both dimensions geometrically influence calculations \phi(j, t)
            phi_combined = torch.sqrt(phi_mb_val * bar_phi_hw)
            
            # Translate directly directly logically into \alpha
            alpha_j = 1.0 - phi_combined
            alpha_scores[j] = alpha_j
            
        return alpha_scores
