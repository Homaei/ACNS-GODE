import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple

class SubgraphBoundaryEvaluator:
    """
    Log-Normalized Gap Criterion mathematically identifying the true edge of a topological 
    cyber-attack. Cross-verified rigorously by Community Conductance (\Phi(S)) logic.
    """
    def __init__(self, eps_ppr: float = 1e-10, eps: float = 1e-12, phi_max: float = 0.3):
        """
        Initialization constraints.
        
        Args:
            eps_ppr (float): Floor establishing meaningful mathematical traversal mass.
            eps (float): Strictly prevents log-domain zeroes.
            phi_max (float): The expansion threshold \Phi_{max} limit for subgraph coherence.
        """
        self.eps_ppr = eps_ppr
        self.eps = eps
        self.phi_max = phi_max

    def _log_normalized_gap(self, sorted_probs: np.ndarray, n: int) -> np.ndarray:
        """
        Calculates dimensionless gap criterion Eq 16.
        """
        gaps = np.zeros(n - 1)
        
        max_pi = max(sorted_probs[0], self.eps_ppr)
        min_pi = max(sorted_probs[n-1], self.eps_ppr)
        denom = np.log(max_pi) - np.log(min_pi) + self.eps
        
        for k in range(n - 1):
            if sorted_probs[k] <= self.eps_ppr:
                break
            
            p_k = max(sorted_probs[k], self.eps_ppr)
            p_k1 = max(sorted_probs[k+1], self.eps_ppr)
            
            num = np.log(p_k) - np.log(p_k1)
            gaps[k] = num / denom
            
        return gaps

    def _compute_conductance(self, graph: nx.DiGraph, S: Set[int]) -> float:
        """
        Evaluates physical coherence via NetworkX volumetric cuts (Eq 18).
        """
        cut_size = nx.cut_size(graph, S)
        
        vol_S = sum(dict(graph.degree(S)).values())
        
        S_bar = set(graph.nodes()) - S
        vol_S_bar = sum(dict(graph.degree(S_bar)).values())
        
        denominator = min(vol_S, vol_S_bar)
        
        if denominator == 0:
            return 1.0 # Max isolation penalty
            
        return cut_size / denominator

    def calculate_optimal_cut(self, graph: nx.DiGraph, ppr_scores: Dict[int, float]) -> Set[int]:
        """
        Governing spatial loop determining the active boundary extraction line.
        
        Args:
            graph (nx.DiGraph): The dynamically constructed network adjacency.
            ppr_scores (Dict[int, float]): Normalized mapping of spatial likelihoods.
            
        Returns:
            Set[int]: Exact array of globally evaluated anomaly G_{sub} nodes.
        """
        n = len(graph.nodes())
        if n < 2:
            return set(graph.nodes())
            
        # Reverse structural sort 
        sorted_nodes = sorted(ppr_scores.keys(), key=lambda node: ppr_scores[node], reverse=True)
        sorted_probs = np.array([ppr_scores[node] for node in sorted_nodes])
        
        # Step 1: Base Candidate identification
        gaps = self._log_normalized_gap(sorted_probs, n)
        
        # Fallback Check if the distribution is completely flat
        eps_gap = 1e-6
        if np.max(gaps) < eps_gap:
            # Fallback Eq 19 cumulative extraction mapped to 90%
            cumulative = np.cumsum(sorted_probs)
            total = cumulative[-1]
            k_star = np.where(cumulative / total >= 0.90)[0][0] + 1
        else:
            k_star = np.argmax(gaps) + 1
            
        # Base candidate logic subset
        S = set(sorted_nodes[:k_star])
        
        # Step 2: Conductance Physical Expansion Evaluation 
        phi_S = self._compute_conductance(graph, S)
        
        if phi_S >= self.phi_max:
            # Expand greedily towards remaining highly-probable neighbors
            max_additions = int(np.floor(0.2 * n))
            added = 0
            
            for k in range(k_star, n):
                if added >= max_additions:
                    break
                    
                S.add(sorted_nodes[k])
                added += 1
                
                if self._compute_conductance(graph, S) < self.phi_max:
                    break
                    
        return S
