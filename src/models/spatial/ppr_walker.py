import networkx as nx
import numpy as np
from typing import List, Set, Dict

class PPRSubgraphWalker:
    """
    Extrapolates spatially anomalous bounds by launching Personalized PageRank walks
    from highly anomalous seed nodes (V_restart). Resolves isolation problems during
    highly stealthy multi-point and replay attacks.
    """
    def __init__(self, alpha_ppr: float = 0.85):
        """
        Initialization
        
        Args:
            alpha_ppr (float): Primary damping factor regulating graph penetration depth (Figure 4).
        """
        self.alpha_ppr = alpha_ppr

    def _extract_connected_components(self, graph: nx.DiGraph, restart_nodes: Set[int]) -> List[Set[int]]:
        """
        Resolves multi-point attacks. Splits disconnected seeds so walk energy doesn't dilute
        between entirely disjoint hydraulic locations.
        """
        induced_subgraph = graph.subgraph(restart_nodes)
        components = list(nx.weakly_connected_components(induced_subgraph))
        return components

    def propagate(self, active_graph: nx.DiGraph, restart_set: Set[int]) -> Dict[int, float]:
        """
        Launches PPR mathematically across the hydraulic topology.
        
        Args:
            active_graph (nx.DiGraph): The dynamically constructed adjacency representation.
            restart_set (Set[int]): Strict collection of V_restart nodes surpassing EVT filters.
            
        Returns:
            Dict[int, float]: Ranked nodal probabilities dictating the ultimate subgraph layout.
        """
        if not restart_set:
            return {}
            
        components = self._extract_connected_components(active_graph, restart_set)
        
        global_scores = {node: 0.0 for node in active_graph.nodes()}
        
        for comp in components:
            personalization = {node: 1.0 if node in comp else 0.0 for node in active_graph.nodes()}
            
            # Execute PageRank using inverse probability walks
            try:
                ppr_scores = nx.pagerank(active_graph, 
                                         alpha=self.alpha_ppr, 
                                         personalization=personalization, 
                                         max_iter=100, 
                                         weight='weight')
            except nx.PowerIterationFailedConvergence:
                # Absolute mathematical failsafe
                ppr_scores = personalization
                
            # Aggregate maximal scores safely across distinct disjoint attack zones
            for node, score in ppr_scores.items():
                global_scores[node] = max(global_scores[node], score)
                
        return global_scores
