import os
import pandas as pd
import matplotlib.pyplot as plt

data_root = os.environ['GAMMA_DATA_ROOT']
path = os.path.join(data_root, 'Results', 'CapsuleSans', 'TSNE')

results = {
    'mutag_caps_tsne': {
        'file_path': os.path.join(path, 'mutag_caps_tsne.csv'),
        'title': 'Capsule Hidden Layer',
        'scale': 2,
    },
    'mutag_cnn_tsne': {
        'file_path': os.path.join(path, 'mutag_cnn_tsne.csv'),
        'title': 'Convolutional Layer',
        'scale': 1,
    },
    'mutag_patchy_tsne': {
        'file_path': os.path.join(path, 'mutag_patchy_tsne.csv'),
        'title': 'Tensor Representation',
        'scale': 1,
    },
    'PROTEINS_caps_tsne': {
        'file_path': os.path.join(path, 'PROTEINS_caps_tsne.csv'),
        'title': 'Capsule Hidden Layer',
        'scale': 2,
    },
    'PROTEINS_cnn_tsne': {
        'file_path': os.path.join(path, 'PROTEINS_cnn_tsne.csv'),
        'title': 'Convolutional Layer',
        'scale': 1,
    },
    'PROTEINS_patchy_tsne': {
        'file_path': os.path.join(path, 'PROTEINS_patchy_tsne.csv'),
        'title': 'Tensor Representation',
        'scale': 1,
    },
}

for key, res in results.items():
    df = pd.read_csv(res['file_path'], sep=',', index_col=0)
    pos = df.loc[df['label'] == 1]
    neg = df.loc[df['label'] != 1]

    plt.figure(figsize=(3.5 * res['scale'], 3.5 * res['scale']))
    plt.title(res['title'])
    plt.xlabel('1st t-SNE Dimension')
    plt.ylabel('2nd t-SNE Dimension')

    plt.plot(pos['comp_1'], pos['comp_2'], '.', label='pos')
    plt.plot(neg['comp_1'], neg['comp_2'], '.', label='neg')
    plt.legend()
    plt.tight_layout()

    plt.savefig(key + '.pdf')
    # plt.show()