# coding: utf-8

import os
import random
from copy import copy
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import networkx as nx

try:
    import pynauty as nauty
except:
    print('Pynauty is not installed in this environment')
from scipy.sparse import coo_matrix
from DropboxLoader import DropboxLoader


# ### Load data
'''
#Node id is made to start from 0 due to nauty package requirement, even if it starts from 1 in the original
#Graph id is starting from 1
'''

DIR_PATH = os.environ['GAMMA_DATA_ROOT']+'Samples/'
GRAPH_RELABEL_NAME = '_relabeled'

def get_subset_adj(df_adj, df_node_label,graph_label_num):
    df_glabel = df_node_label[df_node_label.graph_ind == graph_label_num ]
    index_of_glabel = (df_adj['to'].isin(df_glabel.node) & df_adj['from'].isin(df_glabel.node))
    return df_adj[index_of_glabel]


def dfadj_to_dict(df_adj):
    '''
    input: edges and labels
    output: dictionary. key is node_id and value is list of nodes which the node_id connects to with edges.
    '''
    unique_nodes = np.unique( df_adj['from'].unique().tolist() + df_adj['to'].unique().tolist())
    graph =defaultdict(list)
    for key in unique_nodes:
        graph[key] += df_adj.loc[df_adj['from']==key]['to'].values.tolist()
        # graph[key] += df_adj.loc[df_adj['to']==key]['from'].values.tolist()
    return graph


class GraphConverter(object):

    def __init__(self,dataset_name, receptive_field, file_to_save=''):
        # Parameters :
        self.dataset_name = dataset_name
        self.receptive_field = receptive_field
        # Import the data
        self.import_graph_data()
        # Generates the file path to dropbox

        self.width = int(np.ceil(self.get_average_num_nodes()))
        self.generate_file_path(file_to_save)

    def generate_file_path(self,save_name=''):
        dir_path = os.path.join(DIR_PATH, self.dataset_name)
        tensor_file_name = '{}_patchy_tensor_w_{}{}'.format(self.dataset_name,self.width,save_name)
        self.file_path_save = os.path.join(dir_path, tensor_file_name)
        self.file_path_load = '{}.npy'.format(self.file_path_save)

    def check_if_tensor_exists(self):
        return os.path.isfile(self.file_path_load)


    def import_graph_data(self):
        graph_data = DropboxLoader(self.dataset_name)
        self.df_adj = graph_data.get_adj()
        self.df_node_label = graph_data.get_node_label()
        self.df_node_label = pd.concat([self.df_node_label,
                                        graph_data.get_graph_ind()['graph_ind']],
                                        axis=1)

        self.graph_ids = self.df_node_label['graph_ind'].unique()
        self.num_graphs = len(self.graph_ids)

        self.feature_list = self.df_node_label['label'].unique()
        self.num_features = len(self.feature_list)
        # Generating dictionary of graphs:
        self.adj_dict_by_graphId = self.create_adj_dict_by_graphId()

    def relabel_nodes(self):

        list_new_graphs = []
        self.df_node_label_old = copy(self.df_node_label)
        for i in self.graph_ids:
            current_graph = self.df_node_label[self.df_node_label['graph_ind'] == i]
            random.shuffle(current_graph.node.values)
            list_new_graphs.append(current_graph)

        self.df_node_label = pd.concat(list_new_graphs)

    def relabel_edge_list(self):
        self.df_ajd_old = copy(self.df_adj)
        relabel_dict = dict(
            pd.merge(self.df_node_label, self.df_node_label_old, left_index=True, right_index=True).loc[:, ['node_x', 'node_y']].values)
        self.df_adj.replace(relabel_dict, inplace = True)

    def relabel_graphs(self):
        self.relabel_nodes()
        self.relabel_edge_list()
        self.generate_file_path(GRAPH_RELABEL_NAME)

    def get_smallest_node_id_from_adj(self, df_adj):
        return min(df_adj['to'].min(), df_adj['from'].min())

    def create_adj_dict_by_graphId(self):
        '''
        input: df_node_label
        ##return: {1: {0:[0,2,5]}} = {graphId: {nodeId:[node,node,node]}}
        output: graphID and the adj matrix corresponding to that graph
        '''
        adj_dict_by_graphId = {}

        for l in self.graph_ids:
            df_subset_adj = get_subset_adj(self.df_adj, self.df_node_label, graph_label_num=l)
            smallest_node_id = self.get_smallest_node_id_from_adj(df_subset_adj)
            df_subset_adj -= smallest_node_id
            adj_dict_by_graphId[l] = df_subset_adj
        return adj_dict_by_graphId

    def create_adj_coomatrix_by_graphId(self, adj_dict_by_graphId):
        """
        return: a coomatrix per graphId
        """

        adj_coomatrix_by_graphId = {}
        #unique_graph_labels = self.df_node_label.graph_ind.unique()
        unique_graph_labels = self.graph_ids
        for l in unique_graph_labels:
            df_subset_adj = adj_dict_by_graphId[l]
            df_subset_node_label = self.df_node_label[self.df_node_label['graph_ind']== l]
            adjacency = coo_matrix((np.ones(len(df_subset_adj)),
                                    (df_subset_adj.iloc[:, 0].values, df_subset_adj.iloc[:, 1].values)),
                                   shape=(len(df_subset_node_label), len(df_subset_node_label))
                                   )
            adj_coomatrix_by_graphId[l] = adjacency
        return adj_coomatrix_by_graphId

    def canonical_labeling(self,adj_dict_by_graphId):
        all_canonical_labels = []
        for l in self.graph_ids:
            df_subset_adj = adj_dict_by_graphId[l]
            df_subset_nodes = self.df_node_label[self.df_node_label.graph_ind == l]
            # temp_graph_dict = utils.dfadj_to_dict(df_subset_adj)
            temp_graph_dict = dfadj_to_dict(df_subset_adj)
            nauty_graph = nauty.Graph(len(temp_graph_dict), adjacency_dict=temp_graph_dict)
            canonical_labeling = nauty.canonical_labeling(nauty_graph)
            # canonical_labeling = [df_subset_nodes.label.values[i] for i in canonical_labeling]  ###
            all_canonical_labels += canonical_labeling
        return all_canonical_labels

    def get_neighbor_matrix(self, adj_coomatrix_by_graphId):

        """
        Input :
        return: A 2D matrix : (Node number) x (RECEPTIVE_FIELD_SIZE_K).
        """

        for l_ind, l in enumerate(self.graph_ids):
            adjacency = adj_coomatrix_by_graphId[l]
            graph = nx.from_numpy_matrix(adjacency.todense())

            # Create the neighbors with -1 for neighbor assemble.
            # After this, if the RECEPTIVE_FIELD_SIZE_K exceeds the number of WIDTH_W, then fill them with -1
            neighborhoods = np.zeros((self.width, self.receptive_field), dtype=np.int32)
            neighborhoods.fill(-1)

            df_sequence = self.df_node_label[self.df_node_label.graph_ind == l]
            df_sequence = df_sequence.sort_values(by='cano_label')
            smallest_node_id = df_sequence.node.min()

            # CUT GRAPH BY THRESHOLD of cano_label ''' Top width w elements of V according to labeling  '''
            df_sequence = df_sequence.iloc[:self.width, :]
            df_sequence['node'] = df_sequence.node.values - smallest_node_id

            for i, node in enumerate(df_sequence.node):
                df_shortest = pd.DataFrame.from_dict(nx.single_source_dijkstra_path_length(graph, node),
                                                     orient='index')  #
                df_shortest.columns = ['distance']  #
                df_shortest['node'] = df_shortest.index.values  #
                df_shortest = pd.merge(self.df_node_label, df_shortest, on='node', how='right')  #

                # Sort by distance and then by cano_label
                #df_shortest = df_shortest.sort_values(by=['distance', 'cano_label'])  #

                df_shortest = df_shortest.sort_values(by=['distance'])  #


                df_shortest = df_shortest.iloc[:self.receptive_field, :]  #

                for j in range(0, min(self.receptive_field, len(df_shortest))):
                    neighborhoods[i][j] = df_shortest['node'].values[j] + smallest_node_id
            if l_ind == 0:
                neighborhoods_all = neighborhoods
            else:
                neighborhoods_all = np.r_[neighborhoods_all, neighborhoods]
        return neighborhoods_all

    def neighbor_to_tensor(self, neighborhoods):

        nodes_features = pd.get_dummies(self.df_node_label.label,
                                        columns=self.feature_list,
                                        sparse=True)

        #### Reindex and transporse to get columns of get dummy #########
        nodes_features = nodes_features.T.reindex(self.feature_list).T.fillna(0)
        nodes_features = nodes_features.values
        zero_features_for_padding_at_the_end = np.zeros((1, nodes_features.shape[1]), dtype=float)
        nodes_features = np.r_[nodes_features, zero_features_for_padding_at_the_end]

        i_list = np.reshape(neighborhoods, [-1])
        the_place_of_zero_features_for_padding = i_list.max() + 1
        i_list = np.where(i_list < 0, the_place_of_zero_features_for_padding, i_list)
        ret = nodes_features[i_list]

        ret = np.reshape(ret, (self.num_graphs, self.width * self.receptive_field, self.num_features))
        finalshape = ([self.num_graphs, self.width, self.receptive_field, self.num_features])
        ret = np.reshape(ret, finalshape)

        return ret



    def graphs_to_Patchy_tensor(self):


        if self.check_if_tensor_exists():
            print('{} tensor exists, loading it from Dropbox'.format(self.dataset_name))
            return np.load(self.file_path_load)
        else:
            print('Create dictionary of graphs')

            self.adj_coomatrix_by_graphId = self.create_adj_coomatrix_by_graphId(self.adj_dict_by_graphId)
            print('Canonical Labeling')
            cano_label = self.canonical_labeling(self.adj_dict_by_graphId)
            self.df_node_label = pd.concat([self.df_node_label, pd.Series(cano_label, dtype=int, name='cano_label')], axis=1)
            print('Getting the Neighboors ')
            neighbor_matrix = self.get_neighbor_matrix(self.adj_coomatrix_by_graphId)
            print('Neighboors to Tensor')
            result_tensor = self.neighbor_to_tensor(neighbor_matrix)
            self.patchy_tensor = result_tensor
            np.save(self.file_path_save,result_tensor)
            return result_tensor


    def get_average_num_nodes(self):
        if not hasattr(self,'adj_dict_by_graphId'):
            self.adj_dict_by_graphId = self.create_adj_dict_by_graphId()
        self.nodes_per_graph = []
        for graph_key in self.graph_ids:
            adj_dict = self.adj_dict_by_graphId[graph_key]
            self.nodes_per_graph.append(len(np.unique(adj_dict.values)))
        self.avg_nodes_per_graph = np.mean(self.nodes_per_graph)
        return self.avg_nodes_per_graph
















