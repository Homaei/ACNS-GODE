import torch
import torch.nn as nn
try:
    from torchdiffeq import odeint
except ImportError:
    # Fallback/stub if torchdiffeq is not installed in environment
    odeint = None

from .gat_ode import GraphAttentionODE
from .dynamic_adj import DynamicAdjacency

class ODEIntegrator(nn.Module):
    """
    Wraps the torchdiffeq solver to evaluate the Initial Value Problem directly
    at observed SCADA timestamps, formally overcoming discrete-time interpolation issues.
    """
    def __init__(self, vector_field: GraphAttentionODE, rtol: float = 1e-3, atol: float = 1e-4):
        """
        Args:
            vector_field (GraphAttentionODE): The underlying continuous attention dynamics parameterization.
            rtol (float): Relative error tolerance for the dormand-prince RK45 solver.
            atol (float): Absolute error tolerance.
        """
        super().__init__()
        self.vector_field = vector_field
        self.rtol = rtol
        self.atol = atol
        self.method = 'dopri5' # Dormand-Prince explicitly requested by paper architecture

    def forward(self, z0: torch.Tensor, observation_times: torch.Tensor, dynamic_adj: torch.Tensor) -> torch.Tensor:
        """
        Solves Eq 9: Z(t_i) = Z(t_{i-1}) + \int f_\theta(Z(\tau), A(\tau)) d\tau
        
        Args:
            z0 (torch.Tensor): Initial latent state space representation Z(t_0).
            observation_times (torch.Tensor): Sorted array of timestamps t_0 to t_N (e.g. accounting for packet loss jumps).
            dynamic_adj (torch.Tensor): The pre-computed operational matrix A(t) held mostly constant via Zero-Order Hold.
            
        Returns:
            torch.Tensor: Evaluated trajectory solutions Z(t_i) aligned precisely to `observation_times`.
        """
        if odeint is None:
            raise ImportError("torchdiffeq is required to solve the Graph-ODE dynamics.")
            
        # Due to constraints in torchdiffeq, the vector field is strictly encapsulated.
        # We synthesize an autonomous functional wrapper that holds the static topology constant.
        # (See Algorithm 1: A(tau) assumes Zero-Order Hold during integration steps).
        
        class VectorFieldWrapper(nn.Module):
            def __init__(self, vf, adj):
                super().__init__()
                self.vf = vf
                self.adj = adj
                
            def forward(self, t, z):
                return self.vf(t, z, self.adj)
                
        wrapper = VectorFieldWrapper(self.vector_field, dynamic_adj)
        
        # Integration logic via torchdiffeq library
        z_t = odeint(wrapper, 
                     z0, 
                     observation_times, 
                     rtol=self.rtol, 
                     atol=self.atol, 
                     method=self.method)
                     
        return z_t
