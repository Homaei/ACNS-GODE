import numpy as np
from scipy.stats import genpareto, ks_1samp

class EVTPeaksOverThreshold:
    """
    Implements dynamic statistical thresholding via Extreme Value Theory (EVT)
    to combat sustained threshold poisoning (ramp attacks) and seasonal demand drift.
    Mathematically ensures statistical boundaries are continuously tracked using rolling distributions.
    """
    def __init__(self, W: int = 2016, T_roc: int = 144, delta_max: float = 0.05):
        """
        Initialization of operational windows and constraints.
        
        Args:
            W (int): Rollback window size for historical GPD fitting (approx 1 week).
            T_roc (int): Quick temporal reference for monitoring baseline poisoning (RoC bounds).
            delta_max (float): Predetermined limit \Delta_{max} against aggressive metric drifts.
        """
        self.W = W
        self.T_roc = T_roc
        self.delta_max = delta_max
        self.p_q = 0.999 # Target return probability for strict outlier containment

    def compute_node_threshold(self, error_window: np.ndarray, current_error: float) -> tuple:
        """
        Derives threshold bounds explicitly via GPD distributions fitted mathematically.
        
        Args:
            error_window (np.ndarray): Historical series of nodal reconstruction lengths s_{i,t-W} ...
            current_error (float): Evaluated error logic currently processing.
            
        Returns:
            tuple: (Evaluated extreme threshold limit, Status of RoC check (True for Frozen, False for Active))
        """
        if len(error_window) < self.W:
            # During warm-up initialization fallback directly to classical parametric mapping
            mean = np.mean(error_window) if len(error_window) > 0 else 0.0
            std_dev = np.std(error_window) if len(error_window) > 0 else 1.0
            return mean + 3 * std_dev, False
            
        # Extract operational subset
        active_window = error_window[-self.W:]
        
        # Determine empirical quantile baseline Eq 11 
        u_base = np.percentile(active_window, 95)
        
        # Check against poisoning via Rate of Change monitor 
        short_base = np.percentile(active_window[-self.T_roc:], 95)
        roc = abs(u_base - short_base)
        
        is_frozen = roc > self.delta_max
        
        if is_frozen:
            # If poisoned, conservatively revert to the strict older baseline entirely.
            # Here we simulate the return of the pre-calculated frozen threshold logic.
            return u_base * 1.5, True 
            
        # Locate exceedance boundaries \mathcal{E}_{i,t}
        exceedances = active_window[active_window > u_base] - u_base
        
        if len(exceedances) < 50:
            # Degenerate fit logic (fallback to 3-sigma if not enough statistical mass exists)
            return u_base + 3 * np.std(active_window), False
            
        # Fit GPD scale and shape distributions parameters (\xi, \psi) via maximum likelihood
        try:
            params = genpareto.fit(exceedances, floc=0) # Fix location exactly to 0 for pure deviation 
            xi, _, psi = params
            
            # Goodness-of-Fit verification via Kolmogorov-Smirnov test against uniform theoretical curves
            ks_stat, _ = ks_1samp(exceedances, 'genpareto', args=params)
            
            if ks_stat > 0.10: # Reject fit if uncharacteristically noisy
                return u_base + 3 * np.std(active_window), False
                
        except Exception:
            # Optimization non-convergence fallback
            return u_base + 3 * np.std(active_window), False
        
        # Calculate extreme statistical marker boundary (Eq 13)
        F_emp = 1.0 - (len(exceedances) / len(active_window))
        
        if xi == 0:
            threshold = u_base - psi * np.log((1 - self.p_q) / (1 - F_emp))
        else:
            term = (((1 - self.p_q) / (1 - F_emp)) ** (-xi)) - 1
            threshold = u_base + (psi / xi) * term
            
        return threshold, False
