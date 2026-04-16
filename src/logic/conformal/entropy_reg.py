import torch
import torch.nn as nn

class TemperatureEntropyRegularizer:
    """
    Penalizes extreme logic configurations mathematically ensuring temperature \tau limits 
    remain structurally valid strictly against boundaries establishing Eq 34.
    """
    def __init__(self, lambda_tau: float = 1e-3, tau_min: float = 0.01, tau_max: float = 1.0, eps: float = 1e-12):
        """
        Configurations.
        
        Args:
            lambda_tau (float): Constant defining specific global constraint limit \lambda_{\tau}.
            tau_min (float): Structural floor limits mapping fractional variables.
            tau_max (float): Top bound restricting absolute limits.
        """
        self.lambda_tau = lambda_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.eps = eps

    def compute_penalty(self, tau_mb: torch.Tensor, tau_hw: torch.Tensor, tau_pump: torch.Tensor) -> torch.Tensor:
        """
        Validates the unified penalty matrix explicitly minimizing \mathcal{R}_{\tau}.
        
        Args:
            tau_mb (torch.Tensor): Fractional temperatures array limits restricting node operations.
            tau_hw (torch.Tensor): Fractional temperatures across strictly hydraulic pipelines.
            tau_pump (torch.Tensor): Fractional limits over restricted manufacturer implementations.
            
        Returns:
            torch.Tensor: Negative mathematically restricted constant \mathcal{R}_{\tau}.
        """
        # Node-level limits penalty bounds 
        norm_mb = (tau_mb - self.tau_min) / (self.tau_max - self.tau_min) + self.eps
        pen_mb = torch.sum(torch.log(norm_mb))
        
        # Combined edge-level limits penalty bounds
        if len(tau_hw) > 0:
            norm_hw = (tau_hw - self.tau_min) / (self.tau_max - self.tau_min) + self.eps
            pen_hw = torch.sum(torch.log(norm_hw))
        else:
            pen_hw = torch.tensor(0.0)
            
        if len(tau_pump) > 0:
            norm_pump = (tau_pump - self.tau_min) / (self.tau_max - self.tau_min) + self.eps
            pen_pump = torch.sum(torch.log(norm_pump))
        else:
            pen_pump = torch.tensor(0.0)
            
        return -self.lambda_tau * (pen_mb + pen_hw + pen_pump)
