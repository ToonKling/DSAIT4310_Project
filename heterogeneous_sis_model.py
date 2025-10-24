import numpy as np
import pandas as pd
import networkx as nx
import scipy
from scipy import optimize
from scipy.sparse import linalg
from typing import Dict, List, Tuple, Any

class HeterogeneousSISModel:
    """
    Heterogeneous SIS epidemic model for airport congestion analysis.
    
    Implements the model from "Modeling airport congestion contagion by 
    heterogeneous SIS epidemic spreading on airline networks" (Ceria et al., 2021)
    """

    def __init__(self, G: nx.Graph, c: float = 0.02, theta: float = 1.5, 
                beta: float = 1.0, delta_base: float = 1.0):
        """
        Initialize heterogeneous SIS model.
        
        Args:
            G: NetworkX graph (G1, G2, or G3 from network_construction.py)
            c: Recovery rate constant parameter
            theta: Recovery rate heterogeneity parameter  
            beta: Base infection rate
            delta_base: Base recovery rate scaling factor
        """
        self.G = G
        self.c = c
        self.theta = theta
        self.beta = beta
        self.delta_base = delta_base

        # TODO: will be calculated by helper functions (Tasks 1-3)
        self.recovery_rates = {}
        self.epidemic_threshold = 0.0
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Calculate recovery rates and epidemic threshold.
        """
        # TODO: implement calculate_recovery_rates()
        self.recovery_rates = self._calculate_recovery_rates()

        # TODO: implement calculate_epidemic_threshold()
        self.epidemic_threshold = self._calculate_epidemic_threshold()

    def _calculate_recovery_rates(self) -> Dict[str, float]:
        """
        Calculate heterogeneous recovery rates δᵢ = δ(c + (sᵢ/s_max)^θ)
        
        Returns:
            TODO: Dict mapping airport codes to recovery rates
        """
        recovery_rates = {}
        return recovery_rates

    def _calculate_epidemic_threshold(self) -> float:
        """
        Calculate epidemic threshold \\lambda_1 (A*) from paper equation (3).
        
        Returns:
            TODO: Largest eigenvalue of A* matrix
        """
        return 0.5  # Placeholder 

    def _solve_nimfa_steady_state(self, airports : list[str], network, v0 = None) -> dict[str, float]:
        """
        Solve NIMFA equations for meta-stable (funny term) infection probabilities.
        
        Returns:
            TODO: Dict mapping airport codes to infection probabilities
        """
       
        # I cannot guarantee this is correct.
        # Before fixing I'm looking at evaluation so I know if I'm improving something.

        A = nx.adjacency_matrix(network).todense()
        tau = self.beta / self.delta_base

        # Epidemic threshold
        lambda_max = np.real(linalg.eigs(A, k=1, which='LM')[0][0])
        tau_c = 1.0 / lambda_max

        N = A.shape[0]
        if v0 is None:
            v0 = np.full(N, 1e-3)

        def f(v):
            # Avoid division issues
            Av = A @ v
            return v - (tau * Av) / (1 + tau * Av)

        result = optimize.root(f, v0, method='hybr', tol=1e-10)
        v = np.clip(result.x, 0, 1)
        return {n: v[i] for i, n in enumerate(airports)}

    def run_simulation(self) -> Dict[str, float]:
        """
        Run SIS simulation and return predicted vulnerabilities.
        I think this is how it goes?
        
        Returns:
            Dict mapping airport codes to predicted vulnerability values
        """
        # Check if above epidemic threshold
        if self.epidemic_threshold <= self.c:
            # Below epidemic threshold - no persistent infection
            print(f'Below epidemic threshold')
            return {node: 0.0 for node in self.G.nodes()}
        else:
            # Above threshold - find meta-stable state using NIMFA
            airports = self.G.nodes()
            print(f'Above epidemic threshold, airports: {airports}')
            return self._solve_nimfa_steady_state(airports, self.G)

    def _jensen_shannon_divergence(self, actual: List[float], 
                                predicted: List[float], base = None) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions.
        Calculation from paper
        
        Args:
            actual: List of actual vulnerability values
            predicted: List of predicted vulnerability values
            
        Returns:
            TODO: JSD value (lower is better)
            
        
        """
        p = np.asarray(actual)
        q = np.asarray(predicted)
        p = p / np.sum(p, axis=0)
        q = q / np.sum(q, axis=0)
        m = (p + q) / 2.0
        left = scipy.stats.entropy(p, m)
        right = scipy.stats.entropy(q, m)
        js = np.sum(left, axis=0) + np.sum(right, axis=0)
        if base is not None:
            js /= np.log(base)
        return np.sqrt(js / 2.0)


    def _calculate_recognition_quality(self, actual: List[float], 
                                    predicted: List[float]) -> float:
        """
        TODO: Calculate recognition quality ξ from paper equations (5-6).
        
        Args:
            actual: List of actual vulnerability values
            predicted: List of predicted vulnerability values
        """
        return 0.75  # Placeholder 

    def evaluate_performance(self, actual_vulnerabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare model predictions with actual airport vulnerabilities.
        
        Args:
            actual_vulnerabilities: Dict mapping airport codes to actual vulnerability values
        """
        # get model predictions
        predicted_vulnerabilities = self.run_simulation()

        # print(f'Nodes: \n{self.G.nodes()}\n')
        # print(f'predicted vulns: \n{predicted_vulnerabilities}\n')

        # assume actual_vulnerabilities contains all graph nodes - TODO: maybe we can check common_airports
        actual_array = [actual_vulnerabilities[node] for node in self.G.nodes()]
        predicted_array = [predicted_vulnerabilities[node] for node in self.G.nodes()]

        # calculate evaluation metrics (Tasks 4)
        jsd = self._jensen_shannon_divergence(actual_array, predicted_array)
        recognition_quality = self._calculate_recognition_quality(actual_array, predicted_array)

        return { # just returning everything haha
            'jsd': jsd,
            'recognition_quality': recognition_quality,
            'predicted_vulnerabilities': predicted_vulnerabilities,
            'epidemic_threshold': self.epidemic_threshold,
            'above_threshold': self.epidemic_threshold > self.c,
            'num_airports_evaluated': len(self.G.nodes()), # also can check common airports here
            'parameters': {
                'c': self.c,
                'theta': self.theta,
                'beta': self.beta,
                'delta_base': self.delta_base
            }
        }
