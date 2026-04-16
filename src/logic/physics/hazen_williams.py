import torch
import torch.nn as nn

class HazenWilliamsPredicate(nn.Module):
    """
    Evaluates pressure-loss equations structurally against connecting nodes (j, l).
    Utilizes continuous turbulent gating \lambda_{turb}(Q) to prevent non-laminar exploitation.
    """
    def __init__(self, edge_list: list, r_coeffs: dict, eps_laminar: float = 0.005):
        """
        Initialization logic capturing constraints.
        
        Args:
            edge_list (list): Reference mapping for deterministic tuples defining active edge structures in G_{sub}.
            r_coeffs (dict): Extracted constants R_{jl} referencing topography definitions.
            eps_laminar (float): Flow boundary limit determining transition bounds \epsilon_{laminar}.
        """
        super().__init__()
        self.edge_list = edge_list
        self.r_coeffs = r_coeffs
        self.eps_laminar = eps_laminar
        
        self.tau_min = 0.01
        self.tau_max = 1.0
        
        self.tau_raw = nn.Parameter(torch.zeros(len(edge_list)))

    def get_temperatures(self) -> torch.Tensor:
        """
        Resolves continuous bounded mapping constraints identically mapped across nodes.
        """
        sig = torch.sigmoid(self.tau_raw)
        return self.tau_min + (self.tau_max - self.tau_min) * sig

    def forward(self, h_decoded: dict, q_decoded: dict, h_max: dict) -> dict:
        """
        Implements Eq 24 evaluation maps ensuring pressure drops identically traverse resistance curves.
        
        Args:
            h_decoded (dict): Extracted representations establishing \hat{H}_j levels.
            q_decoded (dict): Fluid paths \hat{Q}_{jl} explicitly decoded.
            h_max (dict): Denominators H^{max}_{jl} normalizing equations preventing variable scaling divergence.
            
        Returns:
            dict: Logically smooth evaluations for active topological constraints \phi_{HW}(jl,t) -> [0, 1].
        """
        phi_hw = {}
        temperatures = self.get_temperatures()
        
        eps = 1e-8
        
        for idx, edge in enumerate(self.edge_list):
            j, l = edge
            h_j = h_decoded.get(j, 0.0)
            h_l = h_decoded.get(l, 0.0)
            q_jl = q_decoded.get(edge, 0.0)
            
            r_jl = self.r_coeffs.get(edge, 0.0001)
            
            # Formulate the deterministic physics prior \lambda_{turb}(Q) (Eq 24)
            # Implemented with `.detach()` to serve as a strict gradient-stop operation
            lambda_turb = 1.0 - torch.exp(-torch.abs(torch.tensor(q_jl)) / self.eps_laminar)
            lambda_turb = lambda_turb.detach()
            
            loss_term = r_jl * (torch.abs(torch.tensor(q_jl)) ** 1.852) * torch.sign(torch.tensor(q_jl))
            
            residual = lambda_turb * ((h_j - h_l) - loss_term) / (h_max.get(edge, 1.0) + eps)
            
            tau_jl = temperatures[idx]
            phi_hw[edge] = torch.exp(- (residual ** 2) / (tau_jl ** 2))
            
        return phi_hw
