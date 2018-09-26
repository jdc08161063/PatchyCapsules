# coding: utf-8
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from six.moves import xrange
import pynauty
import time
import networkx as nx
import numpy as np
import pynauty as nauty
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import os
#from DropboxLoader import DropboxLoader

DATA_PATH = '~/.gamma_link/'
DROPBOX_DATA_ROOT = str(DATA_PATH) + 'Samples/'


class Dropbox(object):
    '''
    Node id is made to start from 0 due to nauty package requirement, even if it starts from 1 in the original file
    '''

    def __init__(self, datasetname):
        self.node_label_filename = '{0}/{0}_node_labels.txt'.format(datasetname)
        self.edge_label_filename = '{0}/{0}_edge_labels.txt'.format(datasetname)
        self.adj_filename = '{0}/{0}_A.txt'.format(datasetname)
        self.graph_ind_filename = '{0}/{0}_graph_indicator.txt'.format(datasetname)
        self.graph_label_filename = '{0}/{0}_graph_labels.txt'.format(datasetname)

    def get_node_label(self):
        df_node_label = pd.read_csv(os.path.join(DROPBOX_DATA_ROOT, self.node_label_filename), delimiter=' ', header=None)
        df_node_label.columns = ['label']
        df_node_label['node'] = df_node_label.index.values
        return df_node_label

    def get_edge_label(self):
        df_edge_label = pd.read_csv(os.path.join(DROPBOX_DATA_ROOT, self.edge_label_filename), delimiter=' ', header=None)
        df_edge_label.columns = ['edge_label']
        df_edge_label['edge_id'] = df_edge_label.index.values
        return df_edge_label

    def get_graph_ind(self):
        df_graph_ind = pd.read_csv(os.path.join(DROPBOX_DATA_ROOT, self.graph_ind_filename), delimiter=' ', header=None)
        df_graph_ind.columns = ['graph_ind']
        df_graph_ind['node'] = df_graph_ind.index.values
        return df_graph_ind

    def get_graph_label(self):
        df_graph_ind = pd.read_csv(os.path.join(DROPBOX_DATA_ROOT, self.graph_label_filename), delimiter=' ', header=None)
        df_graph_ind.columns = ['graph_label']
        df_graph_ind['node'] = df_graph_ind.index.values
        return df_graph_ind

    def get_adj(self):
        df_adj = pd.read_csv(os.path.join(DROPBOX_DATA_ROOT, self.adj_filename), delimiter=',', header=None)
        df_adj.columns = ['from', 'to']
        df_adj['from'] = df_adj['from'].values - 1
        df_adj['to'] = df_adj['to'].values - 1
        return df_adj


#from utils import dfadj_to_dict

# ### Load data
'''
#Node id is made to start from 0 due to nauty package requirement, even if it starts from 1 in the original
#Graph id is starting from 1
'''

def dfadj_to_dict(df_adj):
    '''
    input: edges and labels
    output: dictionary. key is node_id and value is list of nodes which the node_id connects to with edges.
    '''
    unique_nodes = np.unique( df_adj['from'].unique().tolist() + df_adj['to'].unique().tolist())
    graph =defaultdict(list)
    for key in unique_nodes:
        graph[key] += df_adj.loc[df_adj['from']==key]['to'].values.tolist()
        graph[key] += df_adj.loc[df_adj['to']==key]['from'].values.tolist()
    return graph

# ### Functions


def get_subset_adj(df_adj, df_node_label,graph_label_num):
    df_glabel = df_node_label[df_node_label.graph_ind == graph_label_num ]
    index_of_glabel = (df_adj['to'].isin(df_glabel.node) & df_adj['from'].isin(df_glabel.node))
    return df_adj[index_of_glabel]

def get_smallest_node_id_from_adj(df_adj):
    return min(df_adj['to'].min(), df_adj['from'].min())


def create_adj_dict_by_graphId(df_adj, df_node_label):
    '''
    input: df_node_label
    ##return: {1: {0:[0,2,5]}} = {graphId: {nodeId:[node,node,node]}}
    output: graphID and the adj matrix corresponding to that graph
    '''
    adj_dict_by_graphId ={}
    unique_graph_labels = df_node_label.graph_ind.unique()
    for l in unique_graph_labels:
        df_subset_adj = get_subset_adj(df_adj, df_node_label, graph_label_num=l)
        smallest_node_id = get_smallest_node_id_from_adj(df_subset_adj)
        df_subset_adj -= smallest_node_id
        adj_dict_by_graphId[l] = df_subset_adj
    return adj_dict_by_graphId


def canonical_labeling(adj_dict_by_graphId, df_node_label, df_adj):
    all_canonical_labels =[]
    unique_graph_labels = df_node_label.graph_ind.unique()
    for l in unique_graph_labels:
        df_subset_adj = adj_dict_by_graphId[l]
        df_subset_nodes = df_node_label[df_node_label.graph_ind==l]
        #temp_graph_dict = utils.dfadj_to_dict(df_subset_adj)
        temp_graph_dict = dfadj_to_dict(df_subset_adj)
        nauty_graph = nauty.Graph(len(temp_graph_dict), adjacency_dict=temp_graph_dict)
        canonical_labeling = nauty.canonical_labeling(nauty_graph)
        canonical_labeling = [df_subset_nodes.label.values[i] for i in canonical_labeling] ###
        all_canonical_labels += canonical_labeling
    return all_canonical_labels


def create_adj_coomatrix_by_graphId(adj_dict_by_graphId, df_node_label):
    """
    return: a coomatrix per graphId
    """

    adj_coomatrix_by_graphId ={}
    unique_graph_labels = df_node_label.graph_ind.unique()
    for l in unique_graph_labels:
        df_subset_adj = adj_dict_by_graphId[l]
        df_subset_node_label = df_node_label[df_node_label.graph_ind == l]
        adjacency = coo_matrix(( np.ones(len(df_subset_adj)),
                                (df_subset_adj.iloc[:,0].values, df_subset_adj.iloc[:,1].values) ),
                                 shape=(len(df_subset_node_label), len(df_subset_node_label))
                              )
        adj_coomatrix_by_graphId[l]=adjacency
    return adj_coomatrix_by_graphId

def make_neighbor(adj_coomatrix_by_graphId, df_node_label, WIDTH_W, RECEPTIVE_FIELD_SIZE_K):

    """
    return: a dictionary with the shape of {graphId:[matrix: node x neighbor]}
    The size of 2D matrix is (Node number) x (RECEPTIVE_FIELD_SIZE_K).
    """
    neighborhoods_dict=dict()
    unique_graph_labels = df_node_label.graph_ind.unique()
    for l_ind, l in enumerate(unique_graph_labels):
        adjacency = adj_coomatrix_by_graphId[l]
        graph = nx.from_numpy_matrix(adjacency.todense())

        # Create the neighbors with -1 for neighbor assemble.
        #After this, if the RECEPTIVE_FIELD_SIZE_K exceeds the number of WIDTH_W, then fill them with -1
        neighborhoods = np.zeros((WIDTH_W, RECEPTIVE_FIELD_SIZE_K), dtype=np.int32)
        neighborhoods.fill(-1)

        df_sequence = df_node_label[df_node_label.graph_ind == l]
        df_sequence = df_sequence.sort_values(by='cano_label')
        smallest_node_id = df_sequence.node.min()

        # CUT GRAPH BY THRESHOLD of cano_label ''' Top width w elements of V according to labeling  '''
        df_sequence = df_sequence.iloc[:WIDTH_W,:]
        df_sequence['node'] = df_sequence.node.values  - smallest_node_id

        for i, node in enumerate(df_sequence.node):
            #shortest = nx.single_source_dijkstra_path_length(graph, node).items()
            df_shortest = pd.DataFrame.from_dict(nx.single_source_dijkstra_path_length(graph, node),
                                                 orient='index') #
            df_shortest.columns =['distance'] #
            df_shortest['node'] = df_shortest.index.values #
            df_shortest = pd.merge(df_node_label, df_shortest, on='node', how='right') #

            # Sort by distance and then by cano_label
            df_shortest = df_shortest.sort_values(by=['distance','cano_label']) #
            df_shortest = df_shortest.iloc[:RECEPTIVE_FIELD_SIZE_K,:] #
            #shortest = sorted(shortest, key=lambda v: v[1])
            #shortest = shortest[:RECEPTIVE_FIELD_SIZE_K]
            for j in range(0, min(RECEPTIVE_FIELD_SIZE_K, len(df_shortest))):
                #neighborhoods[i][j] = shortest[j][0]
                neighborhoods[i][j] = df_shortest['node'].values[j] + smallest_node_id

        #neighborhoods_dict[l]= neighborhoods.copy()
        if l_ind ==0: neighborhoods_all = neighborhoods
        else: neighborhoods_all = np.r_[neighborhoods_all, neighborhoods]
    return neighborhoods_all



# ###### Show the frequency of labels to make threshold
'''
# #How to select top w elements of V according to labeling  
df_node_label.cano_label.value_counts().plot(kind='bar')
df_node_label.cano_label.value_counts().sort_index().plot(kind='bar',  figsize=(14,5))
plt.title('Number of nodes by labeling')
plt.xlabel('Labeling')
plt.ylabel('Number of nodes')

_SUM_ALL_NODES = df_node_label.shape[0]
plt.twinx()
plt.ylabel("Cummlative Sum Rate", color="r")
plt.tick_params(axis="y", labelcolor="r")
plt.plot(df_node_label.cano_label.value_counts().sort_index().index, 
         df_node_label.cano_label.value_counts().sort_index().cumsum() /_SUM_ALL_NODES, "r-", linewidth=2)
plt.show()
'''


# ### Get several nodes with a condition of cano_label (sequence)

def tensor(neighborhoods, WIDTH_W, RECEPTIVE_FIELD_SIZE_K, df_node_label):
    feature_list = feature_list = df_node_label['label'].unique()
    num_features = len(feature_list)

    nodes_features = pd.get_dummies(df_node_label.label,
                               columns=feature_list,
                               sparse=True )

    #### Reindex and transporse to get columns of get dummy #########
    nodes_features = nodes_features.T.reindex(feature_list).T.fillna(0)
    nodes_features = nodes_features.values
    zero_features_for_padding_at_the_end = np.zeros((1, nodes_features.shape[1]), dtype=float)
    nodes_features = np.r_[nodes_features, zero_features_for_padding_at_the_end]

    i_list = np.reshape(neighborhoods, [-1])
    the_place_of_zero_features_for_padding = i_list.max() + 1 
    i_list = np.where(i_list<0 , the_place_of_zero_features_for_padding, i_list)
    ret = nodes_features[i_list]

    graph_data_size = len(df_node_label.graph_ind.unique())

    ret_by_graph = np.reshape(ret, (graph_data_size, WIDTH_W*RECEPTIVE_FIELD_SIZE_K, num_features))
    finalshape = ([graph_data_size, WIDTH_W, RECEPTIVE_FIELD_SIZE_K,num_features])
    ret = np.reshape(ret_by_graph, finalshape)
    return ret


#NUM_NODES
#RECEPTIVE_FIELD_SIZE_K = 20 # Receptive Field Size K
#WIDTH_W = 8 #threshold based on canonical label



def main(WIDTH_W, RECEPTIVE_FIELD_SIZE_K, datasetname='MUTAG'):
    mutag = Dropbox(datasetname)
    #mutag = DropboxLoader(datasetname)

    #df_edge_label = mutag.get_edge_label()
    df_adj = mutag.get_adj()
    df_node_label = mutag.get_node_label()
    df_node_label = pd.concat([df_node_label, mutag.get_graph_ind()['graph_ind']], axis=1)


    adj_dict_by_graphId = create_adj_dict_by_graphId(df_adj, df_node_label)
    adj_coomatrix_by_graphId = create_adj_coomatrix_by_graphId(adj_dict_by_graphId, df_node_label)

    cano_label = canonical_labeling(adj_dict_by_graphId, df_node_label, df_adj)
    df_node_label = pd.concat([df_node_label, pd.Series(cano_label, dtype=int, name='cano_label')], axis=1)

    neighborhoods_graph = make_neighbor(adj_coomatrix_by_graphId, df_node_label, WIDTH_W=WIDTH_W, RECEPTIVE_FIELD_SIZE_K=RECEPTIVE_FIELD_SIZE_K)
    result_tensor = tensor(neighborhoods_graph, WIDTH_W, RECEPTIVE_FIELD_SIZE_K, df_node_label)
    return  result_tensor
