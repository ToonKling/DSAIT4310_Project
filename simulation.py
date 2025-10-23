import pandas as pd
from network_analysis import prepare_data
from network_construction import AirportNetworkBuilder
from heterogeneous_sis_model import HeterogeneousSISModel
from analysis_paper_dataset import get_airports_with_vulnerability

flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv')
flight_data = prepare_data(flight_data)
airports = get_airports_with_vulnerability()
airports = dict(zip(airports['IATA'], airports['VULN']))

builder = AirportNetworkBuilder(flight_data)
G1, G2, G3 = builder.build_all_networks()

model = HeterogeneousSISModel(G1)
model.evaluate_performance(airports)

print(f'Build my networks :)')

