import csv
import numpy as np
import matplotlib.pyplot as plt

def string_to_array(string):
    return np.array(tuple(float(entry) for entry in string[1:-1].split(',')))

def heatmap(data, x_labels, y_labels, title='', cmap=None):
    fig, ax = plt.subplots(num=title)
    if cmap is not None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xlabel('initial layout')
    ax.set_ylabel('regularization weight')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                        ha="center", va="center", color="w" if data[i, j] > np.min(data) + (np.max(data)-np.min(data))/2 else "k")

    fig.tight_layout()
    plt.savefig(title+'.png')

data = np.empty((7, 6, 5, 4))
stage2_data = np.empty((7, 6, 3))

for i, circles in enumerate(range(7, 14)):
    print(r'\subsection{' + str(circles) + 'x' + str(circles) + '}')
    print(r'\begin{adjustwidth}{-1cm}{}')
    print(r'\begin{center}')
    print(r'\begin{tabular}{p{0.5cm}|p{1cm}|p{1cm}|p{1cm}|p{3cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|}')
    print(r'$\lambda$ & $t_\text{comp}$ & $\mathcal{L}_{rend}$ & $\mathcal{L}_{reg}$ & Pruning & F1 & prec & recall & $J$\\')
    print(r'\hline')
    for j, alpha_lambda in enumerate([0, 0.5, 1, 5, 10, 50]):
        directory = '../data/stemcells/simulated/other_initilization_'+ str(circles) +'x'+ str(circles) +'circles_20/optimized_vertex_lists_1.0_0.01_' + str(alpha_lambda)
        with open(directory + '/info.csv', 'r') as info:
            reader = csv.reader(info, delimiter=',')
            entries = [float(entry) for entry in next(reader)]
            stage2_data[i, j] = entries
        with open(directory + '/info2.csv', 'r') as info:
            reader = csv.reader(info, delimiter=',')
            entries = [float(entry) for entry in next(reader)]
            stage2_data[i, j] += entries
        stage2_data[i, j] *= 0.5
        with open(directory + '/test_info2.csv', 'r') as info:
            reader = csv.reader(info, delimiter=';')
            train = [string_to_array(entry) for entry in next(reader)]
            test = [string_to_array(entry) for entry in next(reader)]

            data[i, j, 0] = (train[2] + test[2]) * 0.5
            data[i, j, 1] = train[0]
            data[i, j, 2] = test[0]
            data[i, j, 3] = train[1]
            data[i, j, 4] = test[1]

        print('\n' +
                r'        \multirow{5}{*}{' + str(alpha_lambda) + '}' + '\n' +
                r'        &' + '\n' +
                r'        \multirow{5}{*}{'+'{:.3f}'.format(stage2_data[i, j, 0]) + '}\n' +
                r'        &' + '\n' +
                r'        \multirow{5}{*}{'+'{:.3f}'.format(stage2_data[i, j, 1]) + '}\n' +
                r'        &' + '\n' +
                r'        \multirow{5}{*}{'+'{:.3f}'.format(stage2_data[i, j, 2]) + '}')
        data[i, j, :, 1] = (data[i, j, :, 0] - data[i, j, :, 2]) / (data[i, j, :, 0] - data[i, j, :, 2] + data[i, j, :, 1])
        data[i, j, :, 2] = (data[i, j, :, 0] - data[i, j, :, 2]) / (data[i, j, :, 0])
        data[i, j, :, 0] = 2* (data[i, j, :, 1] * data[i, j, :, 2])/(data[i, j, :, 1] + data[i, j, :, 2])
        print(r'        &       None          & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*data[i, j, 0]) + '\n' +
                r'        \\ \cline{5-9}' + '\n' +
                r'        &&&&    Neural Net train  & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*data[i, j, 3]) + '\n' +
                r'        \\ \cline{5-9}' + '\n' +
                r'        &&&&    Neural Net test  & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*data[i, j, 4]) + '\n' +
                r'        \\ \cline{5-9}' + '\n' +
                r'        &&&&    Threshold train   & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*data[i, j, 1]) + '\n' +
                r'        \\ \cline{5-9}' + '\n' +
                r'        &&&&    Threshold test   & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*data[i, j, 2]) + '\n' +
                r'        \\\hline')
    print(r'\end{tabular}')
    print(r'\end{center}')
    print(r'\end{adjustwidth}')
    print(r'\newpage')

layouts = ['13x13', '12x12', '11x11', '10x10', '9x9', '8x8', '7x7'][::-1]
lambdas = [0, 0.5, 1, 5, 10, 50][::-1]

print(np.nanmax(data[:, :, 2, 0]))
print(np.max(data[:, :, 4, 0]))
heatmap(stage2_data[:, ::-1, 1].transpose(), layouts, lambdas, 'loss', cmap='Oranges')
heatmap(stage2_data[:, ::-1, 0].transpose(), layouts, lambdas, 'computation time', cmap='Reds')
heatmap(data[:, ::-1, 4, 3].transpose(), layouts, lambdas, 'nn jaccard', cmap='Blues')
heatmap(data[:, ::-1, 4, 0].transpose(), layouts, lambdas, 'nn f1', cmap='Greens')
heatmap(data[:, ::-1, 2, 3].transpose(), layouts, lambdas, 'thresh jaccard', cmap='Blues')
heatmap(data[:, ::-1, 2, 0].transpose(), layouts, lambdas, 'thresh f1', cmap='Greens')
heatmap(data[:, ::-1, 0, 3].transpose(), layouts, lambdas, 'none jaccard', cmap='Blues')
heatmap(data[:, ::-1, 0, 0].transpose(), layouts, lambdas, 'none f1', cmap='Greens')
plt.show()