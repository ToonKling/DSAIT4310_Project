from email.mime import base
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

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

def run_intervention_experiment(budget: float, network, c: float, theta: float, tau: float) -> dict[str,]:
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

    return {'default': results_default,'uniform': results_uniform, 'degree': results_degree, 'degree_exp_1': results_degree_exp, 'degree_exp_2': results_degree_exp_2}


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
 
BEST_NETWORK_NAME = 'G2'
BEST_NETWORK = networks_dict[BEST_NETWORK_NAME]
BEST_C = 0.02
BEST_THETA = 1.5
BEST_TAU = optimal_taus[(BEST_C, BEST_THETA)][BEST_NETWORK_NAME]
RECOVERY_BUDGET = 1.05 # How much the average recovery can increase 

budgets = np.arange(1, 1.2, 0.02)
plot_data = []
for budget in budgets:
    results = run_intervention_experiment(budget, BEST_NETWORK, BEST_C, BEST_THETA, BEST_TAU)
    plot_data.append(results)

plot_data = pd.DataFrame(plot_data)

plot_results_varying_budget(budgets, plot_data)
