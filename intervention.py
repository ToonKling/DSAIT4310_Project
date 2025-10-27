import pandas as pd
import matplotlib.pyplot as plt
from heterogeneous_sis_model import HeterogeneousSISModel
from network_analysis import prepare_data
from network_construction import AirportNetworkBuilder

flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv')
flight_data = prepare_data(flight_data)

builder = AirportNetworkBuilder(flight_data)
G1, G2, G3 = builder.build_all_networks()

BEST_NETWORK = G2
BEST_C = 0.02
BEST_THETA = 1.5
BEST_TAU = 1 # Did not check

def measure_effect(intervention = False) -> float:
    model = HeterogeneousSISModel(G=BEST_NETWORK, c=BEST_C, theta=BEST_THETA, tau=BEST_TAU)
    if intervention:
        model.intervention_random()
    vulns = model.run_simulation()
    performance = sum(vulns.values()) / len(vulns)
    return performance

results_default = [measure_effect()]
results_random  = [measure_effect(intervention = True) for _ in range(100)]

plt.figure(figsize=(6, 5))
plt.boxplot([results_default, results_random], labels=['Base case', 'Random intervention'])
plt.ylabel(f"Average vulnerability")

plt.show()
