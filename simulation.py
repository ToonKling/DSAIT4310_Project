import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from network_analysis import prepare_data
from network_construction import AirportNetworkBuilder
from heterogeneous_sis_model import HeterogeneousSISModel
from analysis_paper_dataset import get_airports_with_vulnerability
import pickle

def plot_scatter_coupled_JS_ROC_AUC(shannon, ranking_quality, label, ax, colormap = 'viridis',show_c = False, show_theta = True):
    # We scatter first the Shannon, then the ranking quality
    sc = ax.scatter(shannon, ranking_quality, c=label, edgecolors='k',linewidths=0.2,cmap = colormap,label = '_nolegend_')
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0.68, 0.78)

    cbar = plt.colorbar(sc, ax=ax)
    if show_c: cbar.set_label('c', size = 20)
    if show_theta: cbar.set_label(r'$\theta$', size = 20)
    # tau = df[alpha]['Probabilities']['Network%d_1to14_%1.6f_%1.6f' %(n_network,0,0)].columns[1]
    # plt.axvline(jens_shann(df,alpha,n_network,tau,0,0),label = 'homogeneous_SIS',color = 'k',linestyle = ':')
    # plt.hlines(get_auc_inf(df_RR,alpha,n_network,tau,0,0),xmin=0,xmax=0.35,color = 'k',linestyle = ':')
    # ax.ylabel(r'$\xi$',size = 20)
    # ax.xlabel(r'JSD',size = 20)
    # red_circle = [Line2D([0], [0], marker='o', color='w', label='heterogeneous SIS',
    #                     markeredgecolor ='k',),Line2D([0], [0],color='k', linestyle = ':',label=' homogeneous SIS')]

    # plt.legend(handles=red_circle,loc='lower right')

def draw_graphs(jdts, ranking_qualities, picked_cs, picked_thetas):
    title_dic = {1:'a',2:'b',3:'c'}
    fig, axs = plt.subplots(3,2,figsize = (18,20))
    plt.subplots_adjust(hspace = 0.4)
    for n_network, name in enumerate(['G1', 'G2', 'G3']):
        plot_scatter_coupled_JS_ROC_AUC(jdts[name], ranking_qualities[name], picked_cs[name], axs[n_network, 0], show_c = True, show_theta=False)
        plot_scatter_coupled_JS_ROC_AUC(jdts[name], ranking_qualities[name], picked_thetas[name], axs[n_network, 1])

        axs[n_network,0].text(0.03, 0.9, s= '('+title_dic[n_network+1]+'1)', fontsize=20,transform=axs[n_network,0].transAxes)
        axs[n_network,1].text(0.03, 0.9, s= '('+title_dic[n_network+1]+'2)', fontsize=20,transform=axs[n_network,1].transAxes)
    plt.show()
    #print(ranking_qualities)


def calculate_optimal_taus(networks, avg_vuln, cs_thetas):
    G1, G2, G3 = networks

    #cs_thetas = itertools.product(np.array([0.00001]), np.array([2.0]))
    tau_dict = {}
    for c, theta in cs_thetas:
        tau_dict[(c, theta)] = {}
        for name, network in [('G1', G1), ('G2', G2), ('G3', G3)]:
            model = HeterogeneousSISModel(G=network, c=c, theta=theta)
            tau_opt = model.optimize_tau(c, theta, avg_vuln)
            tau_dict[(c, theta)][name] = tau_opt
            print(f'Optimal tau for {name}, {c:.3f}, {theta:.3f}: {tau_opt}')

    return tau_dict


flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv')
flight_data = prepare_data(flight_data)
airports = get_airports_with_vulnerability()
airports = dict(zip(airports['IATA'], airports['VULN']))

builder = AirportNetworkBuilder(flight_data)
G1, G2, G3 = builder.build_all_networks()

# `c` is a control parameter from the paper, see page 11
#cs = np.arange(0, 0.02, 0.005) # TODO change back to 0.02
cs = np.arange(0, 2.4, 0.4) # TODO change back to 0.02

# `theta` is the other control parameter from paper on page 11. In their code it is called `delta` for some reason.
#thetas = np.arange(1, 2.1, 0.1)
thetas = np.arange(0, 2.4, 0.4)
thetas = np.array([0,1,2])
# This is all combinations of our control variables, aka all dots in figure 5.
cs_thetas = itertools.product(cs, thetas)
cs_thetas_copy = itertools.product(cs, thetas)

jsds = {'G1': [], 'G2' : [], 'G3': []}
recognition_qualities = {'G1': [], 'G2' : [], 'G3': []}
picked_cs = {'G1': [], 'G2' : [], 'G3': []}
picked_thetas = {'G1': [], 'G2' : [], 'G3': []}

avg_vuln = np.mean(list(airports.values()))
filename = f'data/optimal_taus_c_{cs.min()}_{cs.max()}_{cs[1]-cs[0]}_th_{thetas.min()}_{thetas.max()}_{thetas[1]-thetas[0]}'.replace('.',',')
CALC_OPTIMAL_TAUS = True
SAVE_OPTIMAL_TAUS = False
if CALC_OPTIMAL_TAUS:
    optimal_taus = calculate_optimal_taus((G1, G2, G3), avg_vuln, cs_thetas_copy) # Takes a looooooong time, do once and save taus
    if SAVE_OPTIMAL_TAUS:
        with open(filename, 'wb') as f:
            pickle.dump(filename, f)
else:
    with open(filename, 'rb') as f:
        optimal_taus = pickle.load(f)

for c, theta in cs_thetas:
    for name, network in [('G1', G1), ('G2', G2), ('G3', G3)]:
        # DETERMINE OPTIMAL delta first and use that! See paper pg 11
        optimal_tau = optimal_taus[(c, theta)][name]
        print(f'Optimal tau: {optimal_tau}')
        start = time.time()
        print(f'Evaluating model for c={c} and theta={theta} on network {name}')
        model = HeterogeneousSISModel(G=network, c=c, theta=theta, tau=optimal_tau)
        eval = model.evaluate_performance(airports)
        jsd, recognition_quality = eval['jsd'], eval['recognition_quality']
        jsds[name].append(jsd)
        recognition_qualities[name].append(recognition_quality)
        picked_cs[name].append(c)
        picked_thetas[name].append(theta)
        print(f'Took {time.time() - start:.3f} seconds')

print(f'JSDS: \n{jsds['G2']}\n')
print(f'Recognition qualities: \n{recognition_qualities['G2']}\n')

draw_graphs(jsds, recognition_qualities, picked_cs, picked_thetas)


