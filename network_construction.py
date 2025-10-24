import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class AirportNetworkBuilder:
    """
    Implementation of the three networks from the paper:
    - G1: Unweighted network (wij = 1 if direct flight exists)
    - G2: Flight frequency weighted (wij = Fij + Fji)
    - G3: Inverse flight time weighted (wij = 1/E[Tij])
    """
    def __init__(self, flight_data):
        self.flight_data = flight_data.copy()
        self.airports = self._get_airports()
        self.G1 = None  
        self.G2 = None  
        self.G3 = None  

    def _get_airports(self):
        origins = set(self.flight_data['ORIGIN'].dropna())
        destinations = set(self.flight_data['DEST'].dropna())
        return sorted(list(origins | destinations))

    def _calculate_flight_time(self, row):
        try: # sometimes conversion fails -- TODO: handle better
            # Convert HHMM format to minutes
            dep_time = int(row['CRS_DEP_TIME'])
            arr_time = int(row['CRS_ARR_TIME'])

            dep_minutes = (dep_time // 100) * 60 + (dep_time % 100)
            arr_minutes = (arr_time // 100) * 60 + (arr_time % 100)

            # Handle overnight flights
            if arr_minutes < dep_minutes:
                arr_minutes += 24 * 60

            return arr_minutes - dep_minutes
        except:
            return np.nan

    def build_g1_unweighted(self):
        print("Building G1 (unweighted network)")
        G1 = nx.Graph()
        G1.add_nodes_from(self.airports)
        routes = self.flight_data[['ORIGIN', 'DEST']].drop_duplicates()

        for _, row in routes.iterrows():
            origin = row['ORIGIN']
            dest = row['DEST']
            if pd.notna(origin) and pd.notna(dest):
                G1.add_edge(origin, dest, weight=1.0)

        self.G1 = G1
        print(f"G1 created: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
        return G1

    def build_g2_frequency_weighted(self):
        """
        Build G2: Flight frequency weighted network where wij = Fij + Fji
        """
        print("Building G2 (flight frequency weighted)")

        flight_counts = defaultdict(int)

        for _, row in self.flight_data.iterrows():
            origin = row['ORIGIN']
            dest = row['DEST']
            if pd.notna(origin) and pd.notna(dest):
                edge_key = tuple(sorted([origin, dest]))
                flight_counts[edge_key] += 1
                
        G2 = nx.Graph()
        G2.add_nodes_from(self.airports)

        # Add weighted edges
        max_weight = 0
        for (airport1, airport2), count in flight_counts.items():
            G2.add_edge(airport1, airport2, weight=count)
            max_weight = max(max_weight, count)

        # Normalize weights to [0, 1] like the paper
        for u, v, data in G2.edges(data=True):
            data['weight'] = data['weight'] / max_weight

        self.G2 = G2
        print(f"G2 created: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
        print(f"Max flight frequency: {max_weight}")
        return G2

    def build_g3_flight_time_weighted(self):
        """
        Build G3: Inverse flight time weighted network where wij = 1/E[Tij]
        """
        print("Building G3 (inverse flight time weighted)")

        flight_data_with_time = self.flight_data.copy()
        # flight_data_with_time['FLIGHT_TIME'] = flight_data_with_time.apply(self._calculate_flight_time, axis=1)
        flight_data_with_time['FLIGHT_TIME'] = (flight_data_with_time['ARR_ACT'] - flight_data_with_time['DEP_ACT']).dt.seconds / 60

        # calculate the time frequency for each pair of airports (edge)
        flight_data_with_time['node1'] =  flight_data_with_time.apply(lambda row: sorted([row['ORIGIN'], row['DEST']])[0], axis=1)
        flight_data_with_time['node2'] =  flight_data_with_time.apply(lambda row: sorted([row['ORIGIN'], row['DEST']])[1], axis=1)
        edges = flight_data_with_time.groupby(['node1', 'node2']).agg(avg_flight_time=('FLIGHT_TIME', 'mean')).reset_index()
        edges = edges.dropna(subset=['avg_flight_time'])

        # calculate the weights (inverse flight time) 
        edges['weight'] = (1/edges['avg_flight_time'])
        # normalize the weights
        max_weight = max(edges['weight'])
        edges['weight'] = edges['weight'] / max_weight
        print(edges['weight'])
        
        G3 = nx.Graph()
        # Add edges with weights to the graph
        for _, row in edges.iterrows():
            a = row['node1']
            b = row['node2']
            w = row['weight']
            G3.add_edge(a, b, weight=w)

        self.G3 = G3
        print(f"G3 created: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")
        print(f"Flight time range: {flight_data_with_time['FLIGHT_TIME'].min():.0f}-{flight_data_with_time['FLIGHT_TIME'].max():.0f} minutes")
        return G3

    def build_all_networks(self):
        CACHE_FILE = "./all_networks.cache"
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        self.build_g1_unweighted()
        self.build_g2_frequency_weighted()
        self.build_g3_flight_time_weighted()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((self.G1, self.G2, self.G3), f)
        return self.G1, self.G2, self.G3


    def get_top_connected_airports(self, n=10):
        networks = {'G1': self.G1, 'G2': self.G2, 'G3': self.G3}

        print(f"\nTOP {n} MOST CONNECTED AIRPORTS:")
        print("="*50)

        for name, G in networks.items():
            if G is None:
                continue

            print(f"\n{name}:")
            degrees = dict(G.degree())
            top_airports = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n]

            for i, (airport, degree) in enumerate(top_airports, 1):
                print(f"  {i:2d}. {airport}: {degree} connections")

# this does not work correctly
# def test_networks():
#     print("🔗 NETWORK CONSTRUCTION DEMONSTRATION")
#     print("=" * 60)

#     # Load data
#     flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv', low_memory=False)

#     # Filter to paper's timeframe
#     date_parts = {
#         'year': flight_data['YEAR'],
#         'month': flight_data['MONTH'],
#         'day': flight_data['DAY_OF_MONTH']
#     }
#     flight_data['DATE'] = pd.to_datetime(date_parts)
#     start_date = pd.Timestamp('2018-07-01')
#     end_date = pd.Timestamp('2018-07-14')
#     flight_data = flight_data[(flight_data['DATE'] >= start_date) & (flight_data['DATE'] <= end_date)]

#     print(f"📊 Dataset: {len(flight_data):,} flights ({start_date.date()} to {end_date.date()})")

#     # Build networks
#     builder = AirportNetworkBuilder(flight_data)
#     G1, G2, G3 = builder.build_all_networks()

#     print("\n🏗️  NETWORK SPECIFICATIONS")
#     print("-" * 40)
#     print("G1: Unweighted - wij = 1 (if direct flight exists)")
#     print("G2: Flight frequency - wij = (Fij + Fji) / max_frequency")
#     print("G3: Inverse flight time - wij = (1/E[Tij]) / max_inverse_time")

#     print("\n📈 NETWORK STATISTICS")
#     print("-" * 40)
#     for name, G in [('G1', G1), ('G2', G2), ('G3', G3)]:
#         weights = [data['weight'] for u, v, data in G.edges(data=True)]
#         print(f"{name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
#         print(f"     Weight range: [{min(weights):.4f}, {max(weights):.4f}]")
#         print()

#     print("🔍 EXAMPLE ROUTES & WEIGHTS")
#     print("-" * 40)
#     test_routes = [
#         ('ATL', 'CLT', 'Busy domestic route'),
#         ('LAX', 'SFO', 'High-frequency short route'),
#         ('JFK', 'LAX', 'Transcontinental route'),
#         ('ANC', 'SEA', 'Long-distance route')
#     ]

#     for origin, dest, description in test_routes:
#         if G1.has_edge(origin, dest):
#             w1 = G1[origin][dest]['weight']
#             w2 = G2[origin][dest]['weight']
#             w3 = G3[origin][dest]['weight']

#             # Get flight stats for this route
#             route_flights = flight_data[
#                 ((flight_data['ORIGIN'] == origin) & (flight_data['DEST'] == dest)) |
#                 ((flight_data['ORIGIN'] == dest) & (flight_data['DEST'] == origin))
#             ]

#             print(f"{origin}-{dest} ({description}):")
#             print(f"  Flights: {len(route_flights):,}")
#             print(f"  G1 weight: {w1:.3f} (always 1.0)")
#             print(f"  G2 weight: {w2:.3f} (flight frequency)")
#             print(f"  G3 weight: {w3:.3f} (inverse flight time)")
#             print()

#     print("🏆 TOP 5 AIRPORTS BY NODE STRENGTH")
#     print("-" * 40)

#     # Calculate node strengths for each network
#     for name, G in [('G1', G1), ('G2', G2), ('G3', G3)]:
#         strengths = {}
#         for node in G.nodes():
#             strength = sum(data['weight'] for _, _, data in G.edges(node, data=True))
#             strengths[node] = strength

#         top_5 = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:5]
#         print(f"{name} (Node Strength = Σ edge_weights):")
#         for i, (airport, strength) in enumerate(top_5, 1):
#             print(f"  {i}. {airport}: {strength:.2f}")
#         print()

#     print("✅ VALIDATION CHECKS")
#     print("-" * 40)
#     print(f"✓ Same topology: {G1.edges() == G2.edges() == G3.edges()}")
#     print(f"✓ Same airports: {set(G1.nodes()) == set(G2.nodes()) == set(G3.nodes())}")
#     print(f"✓ G1 all weights = 1: {all(d['weight'] == 1.0 for u, v, d in G1.edges(data=True))}")
#     print(f"✓ G2 weights normalized: {max(d['weight'] for u, v, d in G2.edges(data=True)) == 1.0}")
#     print(f"✓ G3 weights normalized: {max(d['weight'] for u, v, d in G3.edges(data=True)) == 1.0}")
#     return builder, G1, G2, G3


# this doesn't run correctly either, check network_analysis.py
# if __name__ == "__main__":
#     flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv', low_memory=False)
#     builder = AirportNetworkBuilder(flight_data)
#     G1, G2, G3 = builder.build_all_networks()
