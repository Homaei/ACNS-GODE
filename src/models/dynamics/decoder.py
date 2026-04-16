import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    """
    Decodes the final continuous-time latent state Z(t_i) into explicit observational
    features (Pressure Head \hat{H} and Flow \hat{Q}). Ensures direct physical
    units are available for the logic grounding stage.
    """
    def __init__(self, hidden_dim: int, output_features: int = 2):
        """
        Args:
            hidden_dim (int): Complexity dimension of the continuous ODE space.
            output_features (int): Typically 2 representing local H and connected Q.
        """
        super().__init__()
        
        # Linear map as defined in Eq 10: \hat{X}_{t_i} = W_{dec} Z(t_i) + b_{dec}
        self.decoder = nn.Linear(hidden_dim, output_features)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Executes explicit feature mapping.
        
        Args:
            z_t (torch.Tensor): Integrated continuous representation Z(t) shape (..., hidden_dim).
            
        Returns:
            torch.Tensor: Explicit output space \hat{X}_t geometry matching ground truth shapes.
        """
        return self.decoder(z_t)
