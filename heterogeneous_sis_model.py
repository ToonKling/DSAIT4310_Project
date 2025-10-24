import numpy as np
import pandas as pd
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import itertools
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


    def _calculate_recognition_quality(self, actual: List[float], predicted: List[float]) -> float:
        """
        Computes recognition quality ξ using r(f) sampled at f ∈ {0.00, 0.05, ..., 1.00}.
        At each f, we compute the recognition rate based on the top f fraction of nodes,
        then numerically integrate r(f) over [0,1] using the trapezoid rule.
        
        Args:
            actual: List of actual vulnerability values
            predicted: List of predicted vulnerability values
            
        Returns:
            float: The recognition quality ξ.
        """
        a = np.asarray(actual, dtype=float)
        p = np.asarray(predicted, dtype=float)
        N = len(a)
        if N == 0:
            return 0.0

        # Stable rankings (descending)
        a_ranked = np.argsort(-a, kind="mergesort")  # actual ranking
        p_ranked = np.argsort(-p, kind="mergesort")  # predicted ranking

        # Fractions f = 0.00, 0.05, ..., 1.00
        f_values = np.arange(0.0, 1.05, 0.05)
        r_values = np.zeros_like(f_values)

        for i, f in enumerate(f_values):
            n_top = int(round(f * N))
            if n_top == 0:
                # Edge case: f = 0 means no nodes selected.
                # We define recognition = 1 when both lists are empty (perfect match),
                # and 0 otherwise (no overlap).
                r_values[i] = 1.0 if np.allclose(p, a) else 0.0
            else:
                top_actual = set(a_ranked[:n_top])
                top_pred = set(p_ranked[:n_top])
                overlap = len(top_actual & top_pred)
                r_values[i] = overlap / float(n_top)

        # Numerical integration over sampled f using trapezoid rule
        xi = np.trapezoid(r_values, f_values)
        return float(xi)


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

        self.draw_graphs()

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
    
    def plot_scatter_coupled_JS_ROC_AUC(self, ax, colormap = 'viridis',show_c = False, show_theta = True):
        ax.scatter([1, 2, 3], [1, 2, 3], edgecolors='k',linewidths=0.2,cmap = colormap,label = '_nolegend_')
        # cbar = plt.colorbar()
        # if show_c: cbar.set_label('c', size = 20)
        # if show_theta: cbar.set_label(r'$\theta$', size = 20)
        # tau = df[alpha]['Probabilities']['Network%d_1to14_%1.6f_%1.6f' %(n_network,0,0)].columns[1]
        # plt.axvline(jens_shann(df,alpha,n_network,tau,0,0),label = 'homogeneous_SIS',color = 'k',linestyle = ':')
        # plt.hlines(get_auc_inf(df_RR,alpha,n_network,tau,0,0),xmin=0,xmax=0.35,color = 'k',linestyle = ':')
        plt.ylabel(r'$\xi$',size = 20)
        plt.xlabel(r'JSD',size = 20)
        # red_circle = [Line2D([0], [0], marker='o', color='w', label='heterogeneous SIS',
        #                     markeredgecolor ='k',),Line2D([0], [0],color='k', linestyle = ':',label=' homogeneous SIS')]

        # plt.legend(handles=red_circle,loc='lower right')

    def draw_graphs(self):
        title_dic = {1:'a',2:'b',3:'c'}
        fig, axs = plt.subplots(3,2,figsize = (18,20))
        plt.subplots_adjust(hspace = 0.4)
        for n_network in range(1,4):
            
            self.plot_scatter_coupled_JS_ROC_AUC(axs[n_network-1, 0])
            self.plot_scatter_coupled_JS_ROC_AUC(axs[n_network-1, 0])

            axs[n_network-1,0].text(0.03, 0.9, s= '('+title_dic[n_network]+'1)', fontsize=20,transform=axs[n_network-1,0].transAxes)
            axs[n_network-1,1].text(0.03, 0.9, s= '('+title_dic[n_network]+'2)', fontsize=20,transform=axs[n_network-1,1].transAxes)
        plt.show()

