import pickle

# Parse optimal_taus and save them as csv
filename = 'data/optimal_taus_c_0,02_2,0_0,02_th_0,0_2,0_0,1' # Path to the saved taus file
with open(filename, 'rb') as f:
    optimal_taus = pickle.load(f)

lines = ['c,theta,network,optimal_tau']
for key, tau_dict in optimal_taus.items():
    c, theta = key
    for net, tau in tau_dict.items():
        lines.append(f'{c},{theta},{net},{tau}')

output_filename = 'data/optimal_taus.csv'
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')