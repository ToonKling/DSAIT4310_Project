import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
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
        try:
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
        flight_data_with_time['FLIGHT_TIME'] = flight_data_with_time.apply(self._calculate_flight_time, axis=1)

        valid_flights = flight_data_with_time.dropna(subset=['FLIGHT_TIME'])
        valid_flights = valid_flights[valid_flights['FLIGHT_TIME'] > 0]

        # Calculate average flight time between each pair of airports
        route_times = {}
        for _, row in valid_flights.iterrows():
            origin = row['ORIGIN']
            dest = row['DEST']
            flight_time = row['FLIGHT_TIME']

            if pd.notna(origin) and pd.notna(dest):
                edge_key = tuple(sorted([origin, dest]))

                if edge_key not in route_times:
                    route_times[edge_key] = []
                route_times[edge_key].append(flight_time)

        G3 = nx.Graph()
        G3.add_nodes_from(self.airports)

        # Add weighted edges (inverse of average flight time)
        max_weight = 0
        for (airport1, airport2), times in route_times.items():
            avg_time = np.mean(times)
            inverse_time = 1.0 / avg_time  # inverse 
            G3.add_edge(airport1, airport2, weight=inverse_time)
            max_weight = max(max_weight, inverse_time)

        # Normalize weights to [0, 1] like the paper
        for u, v, data in G3.edges(data=True):
            data['weight'] = data['weight'] / max_weight

        self.G3 = G3
        print(f"G3 created: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")
        print(f"Flight time range: {valid_flights['FLIGHT_TIME'].min():.0f}-{valid_flights['FLIGHT_TIME'].max():.0f} minutes")
        return G3

    def build_all_networks(self):
        self.build_g1_unweighted()
        self.build_g2_frequency_weighted()
        self.build_g3_flight_time_weighted()
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


def test_networks():
    print("TESTING NETWORK CONSTRUCTION")
    print("="*50)

    flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv', low_memory=False)

    date_parts = {
        'year': flight_data['YEAR'],
        'month': flight_data['MONTH'],
        'day': flight_data['DAY_OF_MONTH']
    }
    flight_data['DATE'] = pd.to_datetime(date_parts)
    start_date = pd.Timestamp('2018-07-01')
    end_date = pd.Timestamp('2018-07-14')
    flight_data = flight_data[(flight_data['DATE'] >= start_date) & (flight_data['DATE'] <= end_date)]

    builder = AirportNetworkBuilder(flight_data)
    G1, G2, G3 = builder.build_all_networks()

    builder.get_top_connected_airports(10)


    return builder, G1, G2, G3


if __name__ == "__main__":
    builder, G1, G2, G3 = test_networks()