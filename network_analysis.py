import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import networkx as nx
from network_construction import AirportNetworkBuilder


def hhmm_to_timedelta(hhmm):
    # NaN check
    if pd.isna(hhmm):
        return pd.NaT
    # Ensure it's treated as an integer
    hhmm = int(hhmm)
    # Extract hours and minutes
    hours = hhmm // 100
    minutes = hhmm % 100
    return pd.Timedelta(hours=hours, minutes=minutes)


def prepare_data(flight_data):
    date_parts = {
        'year': flight_data['YEAR'],
        'month': flight_data['MONTH'],
        'day': flight_data['DAY_OF_MONTH']
    }
    flight_data['DATE'] = pd.to_datetime(date_parts)

    start_date = pd.Timestamp('2018-07-01')
    end_date = pd.Timestamp('2018-07-14')
    flight_data = flight_data[(flight_data['DATE'] >= start_date) & (flight_data['DATE'] <= end_date)]

    flight_data['DEP_SCH_TD'] = flight_data['CRS_DEP_TIME'].apply(hhmm_to_timedelta)
    flight_data['ARR_SCH_TD'] = flight_data['CRS_ARR_TIME'].apply(hhmm_to_timedelta)

    # Calculate scheduled times
    flight_data['DEP_SCH'] = (flight_data['DATE'] + flight_data['DEP_SCH_TD']).dt.floor("h")
    flight_data['ARR_SCH'] = (flight_data['DATE'] + flight_data['ARR_SCH_TD']).dt.floor("h")

    # Calculate actual times (scheduled + delay)
    flight_data['DEP_ACT'] = flight_data['DATE'] + flight_data['DEP_TIME'].apply(hhmm_to_timedelta).dt.floor("h")
    flight_data['ARR_ACT'] = flight_data['DATE'] + flight_data['ARR_TIME'].apply(hhmm_to_timedelta).dt.floor("h")

    return flight_data

def pdf_linear_bins(values, bins=20):
    """Return x (bin centers) and f(x) using linear bins normalized by bin width."""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    N = len(values)
    if N == 0:
        return np.array([]), np.array([])
    x_min, x_max = values.min(), values.max()
    edges = np.linspace(x_min, x_max, bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    pdf = counts / (N * widths)  # density normalized by bin size
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = pdf > 0
    return centers[mask], pdf[mask]

def edge_weights(G, w='weight'):
    return [d.get(w, 1.0) for _, _, d in G.edges(data=True)]

def node_strengths(G, w='weight'):
    if G.is_directed():
        s = dict(G.in_degree(weight=w))
        for n, v in G.out_degree(weight=w):
            s[n] = s.get(n, 0.0) + v
        return list(s.values())
    else:
        return [v for _, v in G.degree(weight=w)]

def distributions_plot(G2, G3, weight_attr='weight', bins=20):
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    # link weight distribution
    for G, label, color in [(G2, r'$G^2$', 'blue'), (G3, r'$G^3$', 'red')]:
        x, y = pdf_linear_bins(edge_weights(G, weight_attr), bins=bins)
        axes[0].scatter(x, y, label=label, color=color, s=25)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('x', fontsize=10)
    axes[0].set_ylabel(r'$f_W(x)$', fontsize=10)
    axes[0].set_title('(a) Link Weight Distribution', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_box_aspect(1)

    # node strengths distribution
    for G, label, color in [(G2, r'$G^2$', 'blue'), (G3, r'$G^3$', 'red')]:
        x, y = pdf_linear_bins(node_strengths(G, weight_attr), bins=bins)
        axes[1].scatter(x, y, label=label, color=color, s=25)
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('x', fontsize=10)
    axes[1].set_ylabel(r'$f_S(x)$', fontsize=10)
    axes[1].set_title('(b) Node Strengths Distribution', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_box_aspect(1)

    plt.show()


if __name__ == "__main__":
    flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv')
    flight_data = prepare_data(flight_data)

    # build the graphs
    builder = AirportNetworkBuilder(flight_data)
    G1, G2, G3 = builder.build_all_networks()
    
    # calculate the linear coeficients between node strength and different centrality measures for all graphs
    graphs = {'G1': G1, 
              'G2': G2, 
              'G3': G3 }

    for name, graph in graphs.items():
        print(f"\nLinear Coefficients for Graph {name}: Node Strength vs. Centrality Measures")
        strength = dict(graph.degree(weight='weight'))

        # compute clusteting coefficient
        clustering = nx.clustering(graph, weight='weight')
        r, p_value = pearsonr(pd.Series(strength), pd.Series(clustering))
        print(f"Clustering Coefficient: r = {r:.4f}")

        # compute betweeness using the reciprocal of the link's weigth
        H = graph.copy()
        for u, v, d in H.edges(data=True):
            w = d.get('weight', 1.0)
            d['inv_weight'] = 1.0 / w if w > 0 else np.inf
        betweenness = nx.betweenness_centrality(H, weight='inv_weight')
        r, p_value = pearsonr(pd.Series(strength), pd.Series(betweenness))
        print(f"Betweenness Centrality: r = {r:.4f}")

        # compute closeness using the reciprocal of the link's weigth
        closeness = nx.closeness_centrality(H, distance='inv_weight')
        r, p_value = pearsonr(pd.Series(strength), pd.Series(closeness))
        print(f"Closeness Centrality: r = {r:.4f}")

        # compute the principal eigenvector component
        eigenvector = nx.eigenvector_centrality(graph, weight='weight')
        r, p_value = pearsonr(pd.Series(strength), pd.Series(eigenvector))
        print(f"Principal Eigenvector Component: r = {r:.4f}")

    
    # plot the link distribution and strength distributions of G2 and G3 (reproduce fig2)
    distributions_plot(G2, G3, weight_attr='weight', bins=20)
