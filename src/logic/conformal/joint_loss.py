import torch
import torch.nn as nn
from typing import Dict, Set

class JointOptimizationLoss(nn.Module):
    """
    Unifies explicit data-driven MSE limits and formal conformal logic penalties structurally via Eq 33.
    Limits \sigma_1 and \sigma_2 geometrically bounds the gradient balancing mapping ensuring optimal
    \rho representations.
    """
    def __init__(self, num_nodes: int):
        """
        Initialization logic establishing uncertainty constraints.
        
        Args:
            num_nodes (int): N variable limit resolving structurally imbalanced matrices.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.sigma_min = 0.01
        
        # Explicit geometric translation limits preventing \sigma_i \to 0 collapse logic
        self.tilde_sigma_1 = nn.Parameter(torch.tensor(1.0))
        self.tilde_sigma_2 = nn.Parameter(torch.tensor(1.0))

    def get_sigmas(self) -> tuple:
        """
        Translate soft parameters explicitly mapping limits geometrically.
        """
        sigma_1 = self.sigma_min + torch.nn.functional.softplus(self.tilde_sigma_1)
        sigma_2 = self.sigma_min + torch.nn.functional.softplus(self.tilde_sigma_2)
        return sigma_1, sigma_2

    def forward(self, mse_term: torch.Tensor, 
                g_sub: Set[int], 
                alpha_scores: Dict[int, torch.Tensor], 
                q_t: Dict[int, float], 
                r_tau: torch.Tensor) -> tuple:
        """
        Explicit integration bounds.
        
        Args:
            mse_term (torch.Tensor): Extracted \tilde{\mathcal{L}}_{MSE} normalized dynamically.
            g_sub (Set[int]): Filter variable identifying structural bounds strictly.
            alpha_scores (Dict[int, torch.Tensor]): Non-conformable evaluations explicitly translated.
            q_t (Dict[int, float]): Explicit ACI boundaries strictly limits.
            r_tau (torch.Tensor): Translated parameter limits establishing entropy variables.
            
        Returns:
            tuple: (total_loss, mse_loss_val, logic_loss_val) strictly evaluated limits.
        """
        sigma_1, sigma_2 = self.get_sigmas()
        
        # Conformal Logic Penalty \mathcal{L}_{logic} structurally imbalanced mapping limits bounds (Eq 31)
        subgraph_size = len(g_sub)
        
        if subgraph_size > 0:
            penalties = []
            for j in g_sub:
                if j in alpha_scores:
                    alpha_val = alpha_scores[j]
                    q_val = torch.tensor(q_t[j])
                    
                    penalty = torch.relu(alpha_val - q_val)
                    penalties.append(penalty)
                    
            if penalties:
                sum_penalty = torch.sum(torch.stack(penalties))
            else:
                sum_penalty = torch.tensor(0.0)
        else:
            sum_penalty = torch.tensor(0.0)
            
        # Structure normalization mapping mathematically defined variables exactly
        l_logic = (self.num_nodes / (subgraph_size + 1.0)) * sum_penalty
        
        # Construct exact generic structural formulations (Eq 33)
        loss_total = (1.0 / (2.0 * (sigma_1 ** 2))) * mse_term + \
                     (1.0 / (2.0 * (sigma_2 ** 2))) * l_logic + \
                     torch.log(sigma_1 * sigma_2) + r_tau
                     
        return loss_total, mse_term.detach(), l_logic.detach()
