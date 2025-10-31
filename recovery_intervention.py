import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import networkx as nx

from heterogeneous_sis_model import HeterogeneousSISModel
from network_analysis import prepare_data
from network_construction import AirportNetworkBuilder

def measure_effect_delta_base(recovery_factors, model) -> float:
    #recovery_factors_list = [v for (k, v) in recovery_factors.items()]
    vulns = model.run_simulation(recovery_factors=recovery_factors)
    performance = sum(vulns.values()) / len(vulns)
    return performance

def distribute_budget(budget, weights) -> list[float]:
    # Normalize weights
    weights = weights / np.sum(weights)
    alpha = len(weights) * (budget - 1.0)
    factors = 1.0 + alpha * weights   # exact average T
    return factors

def run_intervention_experiment(budget: float, network, c: float, theta: float, tau: float, vuln_threshold: float = None) -> dict[str,]:
    n_nodes = len(network.nodes())
    model = HeterogeneousSISModel(G=network, c=c, theta=theta, tau=tau)
    base_factors = np.ones(n_nodes) # No change in recovery rate, base case
    #results_default = [measure_effect_delta_base(recovery_factors=base_factors, model=model)]
    results_default = measure_effect_delta_base(recovery_factors=base_factors, model=model)

    print('Measuring intervention strategy')
    n_nodes = len(network.nodes())

    # Uniform
    uniform_weights = np.ones(n_nodes)
    uniform_rho = distribute_budget(budget, uniform_weights)
    print(np.mean(uniform_rho))
    #results_uniform = [measure_effect_delta_base(recovery_factors=uniform_rho, model=model) for _ in range(100)]
    results_uniform = measure_effect_delta_base(recovery_factors=uniform_rho, model=model)

    # Degree based
    degrees_dict = dict(network.degree(weight=None))
    degrees = list(degrees_dict.values())
    degree_rho = distribute_budget(budget, degrees)
    print(np.mean(degree_rho))
    results_degree = measure_effect_delta_base(recovery_factors=degree_rho, model=model)

    # Degree with exponent 1.5
    alpha = 1.5
    degrees_exp_dict = dict(network.degree(weight=None))
    degrees_exp = [v ** alpha for v in degrees_exp_dict.values()]
    degree_exp_rho = distribute_budget(budget, degrees_exp)
    print(np.mean(degree_exp_rho))
    results_degree_exp = measure_effect_delta_base(recovery_factors=degree_exp_rho, model=model)

    # Degree with exponent 2
    alpha = 2
    degrees_exp_2 = [v ** alpha for v in degrees_exp_dict.values()]
    degree_exp_rho_2 = distribute_budget(budget, degrees_exp_2)
    print(np.mean(degree_exp_rho_2))
    results_degree_exp_2 = measure_effect_delta_base(recovery_factors=degree_exp_rho_2, model=model)

    # Clustering coefficient
    clustering = list(nx.clustering(network, weight='weight').values())
    clustering = [1 / n if n != 0 else 0 for n in clustering]
    clustering_rho = distribute_budget(budget, clustering)
    results_clustering = measure_effect_delta_base(recovery_factors=clustering_rho, model=model)

    # Betweenness Centrality
    H = network.copy()
    for _, _, d in H.edges(data=True):
        w = d.get('weight', 1.0)
        d['inv_weight'] = 1.0 / w if w > 0 else np.inf
    betweenness = list(nx.betweenness_centrality(H, weight='inv_weight').values())
    betweenness_rho = distribute_budget(budget, betweenness)
    results_between = measure_effect_delta_base(recovery_factors=betweenness_rho, model=model)

    # Closeness Centrality
    closeness = list(nx.closeness_centrality(network, distance='weight').values())
    closeness_rho = distribute_budget(budget, closeness)
    results_closeness = measure_effect_delta_base(recovery_factors=closeness_rho, model=model)

    # Principal Eigenvector component
    eigenvector = list(nx.eigenvector_centrality(network, weight='weight').values())
    eigenvector_rho = distribute_budget(budget, eigenvector)
    results_eigenvector = measure_effect_delta_base(recovery_factors=eigenvector_rho, model=model)

    # just use vulnerability
    # compute baseline vulnerabilities under base_factors (no intervention)
    base_vulns = model.run_simulation(recovery_factors=base_factors)
    # preserve node order consistent with model expectations
    vuln_vals = np.array([base_vulns[n] for n in base_vulns.keys()], dtype=float)
    vuln_rho = distribute_budget(budget, vuln_vals)
    results_vulns = measure_effect_delta_base(recovery_factors=vuln_rho, model=model)

    # --- Vulnerability-threshold based intervention ---
    # If a threshold is provided, compute baseline node vulnerabilities and
    # allocate the budget only among nodes whose baseline vulnerability > vuln_threshold.
    results_threshold = None
    if vuln_threshold is not None:

        # create weights: only nodes with vulnerability > threshold keep their vuln as weight
        mask = vuln_vals > float(vuln_threshold)

        degrees_dict = dict(network.degree(weight=None))
        deg_vals = np.array([degrees_dict[n] for n in base_vulns.keys()], dtype=float)
        # alpha = 1.5
        deg_alpha_half = np.power(deg_vals, 1.5)
        # Mask degrees by vulnerability threshold (only nodes exceeding threshold get weight)
        masked_deg_weights = deg_alpha_half * mask.astype(float)

        degree_alpha_half_rho = distribute_budget(budget, masked_deg_weights)
        results_threshold = measure_effect_delta_base(recovery_factors=degree_alpha_half_rho, model=model)
        # print(f"Degree alpha=2 (thresholded) intervention: mean factor {np.mean(degree_alpha_half_rho):.4f}")

    return {
        'default': results_default,
        'uniform': results_uniform,
        'degree': results_degree,
        'degree_exp_1': results_degree_exp,
        'degree_exp_2': results_degree_exp_2,
        'clustering': results_clustering,
        'between': results_between,
        'close': results_closeness,
        'eigenvector': results_eigenvector,
        'threshold': results_threshold,
        'vulnerability': results_vulns
    }


def run_alpha_sweep(budget: float, network, c: float, theta: float, tau: float, alphas: list[float]) -> dict:
    """Run degree-based intervention for multiple alpha exponents.

    Returns a dict mapping alpha -> average vulnerability (float).
    """
    n_nodes = len(network.nodes())
    model = HeterogeneousSISModel(G=network, c=c, theta=theta, tau=tau)

    # degree values (unweighted count)
    degrees = [v for v in dict(network.degree(weight=None)).values()]
    results = {}
    for a in alphas:
        if a == 0:
            # alpha==0 would make all weights 1 -> equivalent to uniform; keep it valid
            deg_weights = np.ones_like(degrees, dtype=float)
        else:
            deg_weights = [float(d) ** a for d in degrees]

        rho = distribute_budget(budget, np.array(deg_weights, dtype=float))
        perf = measure_effect_delta_base(recovery_factors=rho, model=model)
        results[a] = perf

    return results


def plot_alpha_sweep(alpha_results: dict):
    alphas = sorted(alpha_results.keys())
    values = [alpha_results[a] for a in alphas]

    plt.figure(figsize=(7, 4))
    plt.plot(alphas, values, marker='o')
    plt.xlabel('Alpha (degree exponent)')
    plt.ylabel('Average vulnerability')
    plt.title('Degree-intervention performance vs alpha (fixed budget of 1.12)')
    plt.grid(True)
    plt.show()


def run_alpha_distribution_sweep(budget: float, network, c: float, theta: float, tau: float, alphas: list[float]) -> dict:
    """Run model for each alpha and return full vulnerability lists per alpha."""
    results = {}
    degrees = [v for v in dict(network.degree(weight=None)).values()]
    for a in alphas:
        if a == 0:
            deg_weights = np.ones_like(degrees, dtype=float)
        else:
            deg_weights = [float(d) ** a for d in degrees]

        rho = distribute_budget(budget, np.array(deg_weights, dtype=float))
        model = HeterogeneousSISModel(G=network, c=c, theta=theta, tau=tau)
        vulns = model.run_simulation(recovery_factors=rho)
        vals = list(vulns.values()) if vulns else []
        results[a] = vals

    return results


def plot_alpha_distribution(alpha_dist: dict):
    """Plot boxplots of vulnerability distributions for each alpha."""
    alphas = sorted(alpha_dist.keys())
    data = [alpha_dist[a] for a in alphas]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, positions=alphas, widths=0.3, showfliers=False)
    plt.xlabel('Alpha (degree exponent)')
    plt.ylabel('Vulnerability')
    plt.title('Distribution of node vulnerabilities vs Alpha (boxplots)')
    plt.grid(True, axis='y')
    # reduce xticks for readability
    step = max(1, int(len(alphas) / 10))
    plt.xticks(alphas[::step], [f"{a:.1f}" for a in alphas[::step]])
    plt.tight_layout()
    plt.show()


def plot_box(results_dict):
    results_default = results_dict['default']
    results_uniform = results_dict['uniform']
    results_degree = results_dict['degree']

    plt.figure(figsize=(6, 5))
    plt.boxplot([results_default, results_uniform, results_degree], labels=['Base case', 'Uniform intervention', 'Degree intervention'])
    plt.ylabel(f"Average vulnerability")
    plt.show()

def plot_results_varying_budget(budgets, data: pd.DataFrame):
    default_list = data['default']
    uniform_list = data['uniform']
    degree_list = data['degree']
    degree_exp_1 = data['degree_exp_1']
    degree_exp_2 = data['degree_exp_2']
    clustering = data['clustering']
    between = data['between']
    close = data['close']
    eigenvector = data['eigenvector']
    threshold = data['threshold']
    vulnerability = data['vulnerability']

    plt.figure(figsize=(8, 6))
    plt.scatter(budgets, default_list, c='tab:blue', label='Base case', alpha=0.9)
    plt.plot(budgets, default_list, c='tab:blue', alpha=0.4)

    plt.scatter(budgets, uniform_list, c='tab:orange', label='Uniform intervention', alpha=0.9)
    plt.plot(budgets, uniform_list, c='tab:orange', alpha=0.4)

    plt.scatter(budgets, degree_list, c='tab:green', label='Degree intervention', alpha=0.9)
    plt.plot(budgets, degree_list, c='tab:green', alpha=0.4)

    plt.scatter(budgets, degree_exp_1, c='tab:red', label='Degree intervention a = 1.5', alpha=0.9)
    plt.plot(budgets, degree_exp_1, c='tab:red', alpha=0.4)

    plt.scatter(budgets, degree_exp_2, c='tab:purple', label='Degree intervention a = 2', alpha=0.9)
    plt.plot(budgets, degree_exp_2, c='tab:purple', alpha=0.4)

    plt.scatter(budgets, clustering, c='tab:brown', label='Inverse Clustering coefficient', alpha=0.9)
    plt.plot(budgets, clustering, c='tab:brown', alpha=0.4)

    plt.scatter(budgets, between, c='tab:pink', label='Betweenness Centrality', alpha=0.9)
    plt.plot(budgets, between, c='tab:pink', alpha=0.4)

    plt.scatter(budgets, close, c='tab:gray', label='Closeness Centrality', alpha=0.9)
    plt.plot(budgets, close, c='tab:gray', alpha=0.4)

    plt.scatter(budgets, eigenvector, c='tab:olive', label='Principal Eigenvector Component', alpha=0.9)
    plt.plot(budgets, eigenvector, c='tab:olive', alpha=0.4)

    plt.scatter(budgets, threshold, c='tab:cyan', label='Vuln-threshold degree intervention', alpha=0.9)
    plt.plot(budgets, threshold, c='tab:cyan', alpha=0.4)

    plt.scatter(budgets, vulnerability, c='m', label='Vulnerability intervention', alpha=0.9)
    plt.plot(budgets, vulnerability, c='m', alpha=0.4)

    plt.xlabel('Budget')
    plt.ylabel('Average vulnerability')
    plt.title('Intervention performance vs Budget')
    plt.legend()
    plt.grid(True)
    plt.show()



print('Reading flight data...')
flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv')
print('Preparing data...')
flight_data = prepare_data(flight_data)
print('Building networks')
builder = AirportNetworkBuilder(flight_data)
G1, G2, G3 = builder.build_all_networks()
networks_dict = {'G1':G1, 'G2':G2, 'G3': G3}

filename = 'optimal_taus_dict.pickle'
with open(filename, 'rb') as f:
    optimal_taus = pickle.load(f)
 
BEST_NETWORK_NAME = 'G1'
BEST_NETWORK = networks_dict[BEST_NETWORK_NAME]
BEST_C = 0.02
BEST_THETA = 1.5
BEST_TAU = optimal_taus[(BEST_C, BEST_THETA)][BEST_NETWORK_NAME]
RECOVERY_BUDGET = 1.05 # How much the average recovery can increase 

budgets = np.arange(1, 1.2, 0.02)
plot_data = []
for budget in budgets:
    results = run_intervention_experiment(budget, BEST_NETWORK, BEST_C, BEST_THETA, BEST_TAU, vuln_threshold=0.2)
    plot_data.append(results)

plot_data = pd.DataFrame(plot_data)

plot_results_varying_budget(budgets, plot_data)


# # --- Alpha sweep experiment (degree exponent) with fixed budget ---
# intervention_budget = 1.12  # fixed budget requested for alpha sweep
# # choose alphas to test (include small, typical and larger exponents)
# alphas = np.arange(0, 10, 0.5)  # alphas from 0 to 10

# print(f"Running alpha sweep for degree-based intervention with budget={intervention_budget}...")
# alpha_results = run_alpha_sweep(intervention_budget, BEST_NETWORK, BEST_C, BEST_THETA, BEST_TAU, alphas)

# print('Alpha sweep results (alpha -> average vulnerability):')
# for a in sorted(alpha_results.keys()):
#     print(f"  alpha={a}: {alpha_results[a]}")

# plot_alpha_sweep(alpha_results)


# # --- Alpha distribution sweep (boxplots) ---
# print(f"Running alpha distribution sweep (0..10 step 0.5) with budget={intervention_budget}...")
# alpha_dist = run_alpha_distribution_sweep(intervention_budget, BEST_NETWORK, BEST_C, BEST_THETA, BEST_TAU, alphas)
# print('Alpha distribution sweep completed. Showing boxplot...')
# plot_alpha_distribution(alpha_dist)
