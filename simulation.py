import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from network_analysis import prepare_data
from network_construction import AirportNetworkBuilder
from heterogeneous_sis_model import HeterogeneousSISModel
from analysis_paper_dataset import get_airports_with_vulnerability

def plot_scatter_coupled_JS_ROC_AUC(shannon, ranking_quality, ax, colormap = 'viridis',show_c = False, show_theta = True):
    # We scatter first the Shannon, then the ranking quality
    ax.scatter(shannon, ranking_quality, edgecolors='k',linewidths=0.2,cmap = colormap,label = '_nolegend_')

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

def draw_graphs(jdts, ranking_qualities):
    title_dic = {1:'a',2:'b',3:'c'}
    fig, axs = plt.subplots(3,2,figsize = (18,20))
    plt.subplots_adjust(hspace = 0.4)
    for n_network, name in enumerate(['G1', 'G2', 'G3']):
        plot_scatter_coupled_JS_ROC_AUC(jdts[name], ranking_qualities[name], axs[n_network, 0])
        plot_scatter_coupled_JS_ROC_AUC(jdts[name], ranking_qualities[name], axs[n_network, 1])

        axs[n_network,0].text(0.03, 0.9, s= '('+title_dic[n_network+1]+'1)', fontsize=20,transform=axs[n_network,0].transAxes)
        axs[n_network,1].text(0.03, 0.9, s= '('+title_dic[n_network+1]+'2)', fontsize=20,transform=axs[n_network,1].transAxes)
    plt.show()

flight_data = pd.read_csv('data/OnTimePerformance_July2018.csv')
flight_data = prepare_data(flight_data)
airports = get_airports_with_vulnerability()
airports = dict(zip(airports['IATA'], airports['VULN']))

builder = AirportNetworkBuilder(flight_data)
G1, G2, G3 = builder.build_all_networks()


# `c` is a control parameter from the paper, see page 11
cs = np.arange(0.1, 2.02, 0.1) # TODO change back to 0.02

# `theta` is the other control parameter from paper on page 11. In their code it is called `delta` for some reason.
thetas = np.arange(0.1, 2.1, 0.1)

# This is all combinations of our control variables, aka all dots in figure 5.
cs_thetas = itertools.product(cs, thetas)

jsds = {'G1': [], 'G2' : [], 'G3': []}
recognition_qualities = {'G1': [], 'G2' : [], 'G3': []}
for c, theta in cs_thetas:
    for name, network in [('G1', G1), ('G2', G2), ('G3', G3)]:
        start = time.time()
        print(f'Evaluating model for c={c} and theta={theta} on network {name}')
        model = HeterogeneousSISModel(network, c=c, theta=theta)
        eval = model.evaluate_performance(airports)
        jsd, recognition_quality = eval['jsd'], eval['recognition_quality']
        jsds[name].append(jsd)
        recognition_qualities[name].append(recognition_quality)
        print(f'Took {time.time() - start:.3f} seconds')

print(f'JSDS: \n{jsds}\n')
print(f'Recognition qualities: \n{recognition_qualities}\n')

draw_graphs(jsds, recognition_qualities)


