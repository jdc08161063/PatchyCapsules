# coding: utf-8

import os
import random
from copy import copy
from time import time
import sys

import numpy as np
import pandas as pd
import networkx as nx
from ReceptiveFieldMaker import ReceptiveFieldMaker

try:
    import pynauty as nauty
    from pynauty.graph import canonical_labeling
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
GRAPH_RELABEL_NAME = '_relabelled'

def get_subset_adj(df_adj, df_node_label,graph_label_num):
    df_glabel = df_node_label[df_node_label.graph_ind == graph_label_num ]
    index_of_glabel = (df_adj['to'].isin(df_glabel.node) & df_adj['from'].isin(df_glabel.node))
    return df_adj[index_of_glabel]

# Misc function
def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def dfadj_to_dict(df_adj):
    '''
    input: edges and labels
    output: dictionary. key is node_id and value is list of nodes which the node_id connects to with edges.
    '''
    # unique_nodes = np.unique( df_adj['from'].unique().tolist() + df_adj['to'].unique().tolist())
    # graph =defaultdict(list)
    # for key in unique_nodes:
    #     graph[key] += df_adj.loc[df_adj['from']==key]['to'].values.tolist()
    #     # graph[key] += df_adj.loc[df_adj['to']==key]['from'].values.tolist()
    # return graph
    graph = nx.Graph()
    graph.add_edges_from(df_adj.loc[:, ['from','to']].values)
    return nx.to_dict_of_lists(graph)


class PatchyConverter(object):

    def __init__(self, dataset_name, receptive_field, stride, file_to_save='', attr_name='attr_name', label_name='labelling'):
        # Parameters :
        self.dataset_name = dataset_name
        self.receptive_field = receptive_field
        self.k = receptive_field
        self.s = stride
        self.attr_name = attr_name
        self.label_name = label_name
        self.relabel_performed = False
        # Import the data
        print('importing graph data')
        self.import_graph_data()
        # Generates the file path to dropbox
        print('getting width')
        self.width = int(np.ceil(self.get_average_num_nodes()))
        print('width: {}'.format(self.width))
        self.generate_file_path(file_to_save)
        # Times:
        self.times_process_details = {}
        self.times_process_details['normalized_subgraph'] = []
        self.times_process_details['neigh_assembly'] = []
        self.times_process_details['canonicalizes'] = []
        self.times_process_details['compute_subgraph_ranking'] = []
        self.times_process_details['labeling_procedure'] = []
        self.times_process_details['first_labeling_procedure'] = []

    def generate_file_path(self,save_name=''):
        self.dir_path = os.path.join(DIR_PATH, self.dataset_name)
        tensor_file_name = '{}_patchy_tensor_w_{}{}'.format(self.dataset_name,self.width,save_name)
        self.file_path_save = os.path.join(self.dir_path, tensor_file_name)
        self.file_path_load = '{}.npy'.format(self.file_path_save)

        bc_tensor_file_name = '{}_patchy_bc_tensor_w_{}{}'.format(self.dataset_name, self.width, save_name)
        self.file_path_save_bc = os.path.join(self.dir_path, bc_tensor_file_name)
        self.file_path_load_bc = '{}.npy'.format(self.file_path_save_bc)

    def check_if_tensor_exists(self, file):
        return os.path.isfile(file)

    def update_file_path(self,labelling_procedure='bc',save_name=''):
        if labelling_procedure=='bc':
            bc_tensor_file_name = '{}_patchy_bc_tensor_w_{}{}'.format(self.dataset_name, self.width, save_name)
            self.file_path_save = os.path.join(self.dir_path, bc_tensor_file_name)
            self.file_path_load = '{}.npy'.format(self.file_path_save)


    def import_graph_data(self):
        graph_data = DropboxLoader(self.dataset_name)
        self.df_adj = graph_data.get_adj()
        self.df_node_label = graph_data.get_node_label()
        self.df_node_label = pd.concat([self.df_node_label,
                                        graph_data.get_graph_ind()['graph_ind']],
                                        axis=1)

        self.graph_ids = self.df_node_label['graph_ind'].unique()
        self.num_graphs = len(self.graph_ids)
        print('number of graphs in {} dataset : {}'.format(self.dataset_name,self.num_graphs))

        self.feature_list = self.df_node_label['label'].unique()
        self.num_features = len(self.feature_list)
        print('number of features : {}'.format(self.num_features))
        # Generating dictionary of graphs:
        print('Separating Graphs per graph ID')

        self.node_label_by_graphId = self.create_node_label_by_graphId()
        self.adj_dict_by_graphId = self.create_adj_dict_by_graphId()
        self.nx_graphs = self.create_nx_graphs()





    def create_adj_dict_by_graphId(self):
        '''
        input: df_node_label
        ##return: {1: {0:[0,2,5]}} = {graphId: {nodeId:[node,node,node]}}
        output: graphID and the adj matrix corresponding to that graph
        '''
        adj_dict_by_graphId = {}
        node_graph_id = self.df_node_label.loc[:, ['node', 'graph_ind']]
        adj_graph_ind = self.df_adj.merge(node_graph_id, how='inner', left_on='from', right_on='node')
        adj_graph_ind = adj_graph_ind.loc[:, ['from', 'to', 'graph_ind']]
        #self.min_node_per_graph = {}

        for graph_index in self.graph_ids:
            #df_subset_adj = get_subset_adj(self.df_adj, self.df_node_label, graph_label_num=l)
            df_subset_adj = copy(adj_graph_ind[adj_graph_ind['graph_ind']==graph_index])
            #smallest_node_id = self.get_smallest_node_id_from_adj(df_subset_adj)
            #self.min_node_per_graph[graph_index] = smallest_node_id

            df_subset_adj['from'] = df_subset_adj['from'].apply(lambda x: x-self.min_node_per_graph[graph_index])
            df_subset_adj['to'] = df_subset_adj['to'].apply(lambda x: x - self.min_node_per_graph[graph_index])

            adj_dict_by_graphId[graph_index] = df_subset_adj

        return adj_dict_by_graphId


    def create_node_label_by_graphId(self):
        node_label_by_graphId = {}
        self.min_node_per_graph = {}
        for i in self.graph_ids:
            node_label = copy(self.df_node_label[self.df_node_label['graph_ind'] == i])

            min_node_per_graph = node_label.node.min()

            node_label['node'] = node_label['node'].apply(lambda x: x-min_node_per_graph)
            node_label_by_graphId[i] = node_label
            self.min_node_per_graph[i] = min_node_per_graph
        return node_label_by_graphId


    def relabel_graphs(self):
        # Reassign the dictionaries:
        self.adj_dict_by_graphId_old = copy(self.adj_dict_by_graphId)
        self.node_label_by_graphId_old = copy(self.node_label_by_graphId)

        # New dictionaries:
        self.adj_dict_by_graphId = {}
        self.node_label_by_graphId = {}

        # Relabelling:
        for i in self.graph_ids:
            current_node_label_old = self.node_label_by_graphId_old[i]
            current_node_label = copy(current_node_label_old)
            # Shuffle them
            random.shuffle(current_node_label.node.values)
            relabelled_df = pd.DataFrame({'old_node': current_node_label_old.node, 'new_node': current_node_label.node})
            self.node_label_by_graphId[i] = current_node_label

            current_adj_old = self.adj_dict_by_graphId_old[i]
            current_adj = copy(current_adj_old)
            current_adj = current_adj.merge(relabelled_df, how='inner', left_on='from', right_on='old_node').loc[:,
                          ['new_node', 'to', 'graph_ind']]
            current_adj.rename(columns={'new_node': 'from'}, inplace=True)
            current_adj = current_adj.merge(relabelled_df, how='inner', left_on='to', right_on='old_node').loc[:,
                          ['new_node', 'from', 'graph_ind']]
            current_adj.rename(columns={'new_node': 'to'}, inplace=True)
            self.adj_dict_by_graphId[i] = current_adj

        self.generate_file_path(GRAPH_RELABEL_NAME)
        self.relabel_performed = True

    def get_smallest_node_id_from_adj(self, df_adj):
        return min(df_adj['to'].min(), df_adj['from'].min())


    def create_adj_coomatrix_by_graphId(self):
        """
        return: a coomatrix per graphId
        """

        self.adj_coomatrix_by_graphId = {}
        #unique_graph_labels = self.df_node_label.graph_ind.unique()
        unique_graph_labels = self.graph_ids
        for k in unique_graph_labels:
            df_subset_adj = self.adj_dict_by_graphId[k]
            df_subset_node_label = self.node_label_by_graphId[k]
            adjacency = coo_matrix((np.ones(len(df_subset_adj)),
                                    (df_subset_adj.loc[:, 'from'].values, df_subset_adj.loc[:, 'to'].values)),
                                   shape=(len(df_subset_node_label), len(df_subset_node_label))
                                   )
            self.adj_coomatrix_by_graphId[k] = adjacency


    def labelling(self, labelling_procedure='bc'):
        """
        bc : betweenness centrality
        :param labelling_procedure:
        :return: adds label according to labelling procedure
        """

        if labelling_procedure == 'bc':
            self.new_label_name = 'bc_label'
            for k in self.graph_ids:
                nx_graph = self.nx_graphs[k]
                sorted_bc_labels = [i[0] for i in sorted(nx.betweenness_centrality(nx_graph).items(), key=lambda x: x[1], reverse=True)]
                self.node_label_by_graphId[k].reset_index(inplace=True)
                self.node_label_by_graphId[k] = pd.concat([self.node_label_by_graphId[k], pd.Series(sorted_bc_labels, dtype=int, name=self.new_label_name)], axis=1)
        elif labelling_procedure == 'cl':
            self.new_label_name = 'cano_label'
            for j,k in enumerate(self.graph_ids):
                df_subset_adj = self.adj_dict_by_graphId[k]
                temp_graph_dict = dfadj_to_dict(df_subset_adj)
                try:
                    nauty_graph = nauty.Graph(len(temp_graph_dict), adjacency_dict=temp_graph_dict)
                except:

                    missing = self.node_label_by_graphId[k].shape[0] - len(temp_graph_dict.keys())
                    print('missing nodes in graph number {} :  {}'.format(k, missing))
                    nauty_graph = nauty.Graph(len(temp_graph_dict) + missing, adjacency_dict=temp_graph_dict)
                    # raise
                    pass
                canonical_labeling = nauty.canonical_labeling(nauty_graph)
                self.node_label_by_graphId[k].reset_index(inplace=True)
                self.node_label_by_graphId[k] = pd.concat(
                    [self.node_label_by_graphId[k], pd.Series(canonical_labeling, dtype=int, name=self.new_label_name)],
                    axis=1)
                progress_bar(j,self.num_graphs)

    def canonical_labeling(self):

        self.new_label_name = 'cano_label'
        for k in self.graph_ids:
            df_subset_adj = self.adj_dict_by_graphId[k]
            temp_graph_dict = dfadj_to_dict(df_subset_adj)
            try:
                nauty_graph = nauty.Graph(len(temp_graph_dict), adjacency_dict=temp_graph_dict)
            except:

                missing = self.node_label_by_graphId[k].shape[0] - len(temp_graph_dict.keys())
                print('missing nodes in graph number {} :  {}'.format(k,missing))
                nauty_graph = nauty.Graph(len(temp_graph_dict)+missing, adjacency_dict=temp_graph_dict)
                #raise
                pass
            canonical_labeling = nauty.canonical_labeling(nauty_graph)
            self.node_label_by_graphId[k].reset_index(inplace=True)
            self.node_label_by_graphId[k] = pd.concat([self.node_label_by_graphId[k], pd.Series(canonical_labeling, dtype=int, name=self.new_label_name)], axis=1)




    def get_neighbor_matrix(self):

        """
        Input :
        return: A 2D matrix : (Node number) x (RECEPTIVE_FIELD_SIZE_K).
        """
        if not hasattr(self,'adj_coomatrix_by_graphId'):
            self.create_adj_coomatrix_by_graphId()


        for l_ind, l in enumerate(self.graph_ids):
            adjacency = self.adj_coomatrix_by_graphId[l]
            graph = nx.from_numpy_matrix(adjacency.todense())

            # Create the neighbors with -1 for neighbor assemble.
            # After this, if the RECEPTIVE_FIELD_SIZE_K exceeds the number of WIDTH_W, then fill them with -1
            neighborhoods = np.zeros((self.width, self.receptive_field), dtype=np.int32)
            neighborhoods.fill(-1)

            df_sequence = self.node_label_by_graphId[l]
            try:
                df_sequence = df_sequence.sort_values(by=self.new_label_name)
            except KeyError:
                self.canonical_labeling()
            #smallest_node_id = df_sequence.node.min()

            # CUT GRAPH BY THRESHOLD of cano_label ''' Top width w elements of V according to labeling  '''
            df_sequence = df_sequence.iloc[:self.width, :]
            #df_sequence['node'] = df_sequence.node.values - self.min_node_per_graph[l]

            for i, node in enumerate(df_sequence.node):
                try:
                    df_shortest = pd.DataFrame.from_dict(nx.single_source_dijkstra_path_length(graph, node),
                                                     orient='index')  #
                except:
                    print('problems in graph {}, node: {}'.format(l,i))
                    raise
                df_shortest.columns = ['distance']  #
                df_shortest['node'] = df_shortest.index.values  #
                df_shortest = pd.merge(self.node_label_by_graphId[l], df_shortest, on='node', how='right')  #

                # Sort by distance and then by cano_label
                #df_shortest = df_shortest.sort_values(by=['distance', 'cano_label'])  #

                df_shortest = df_shortest.sort_values(by=['distance',self.new_label_name])  #


                df_shortest = df_shortest.iloc[:self.receptive_field, :]  #

                for j in range(0, min(self.receptive_field, len(df_shortest))):
                    neighborhoods[i][j] = df_shortest['node'].values[j] + self.min_node_per_graph[l]
            if l_ind == 0:
                neighborhoods_all = neighborhoods
            else:
                neighborhoods_all = np.r_[neighborhoods_all, neighborhoods]

            progress_bar(l_ind,self.num_graphs)
        self.neighbor_matrix = neighborhoods_all

    def neighbor_to_tensor(self):

        nodes_features = pd.get_dummies(self.df_node_label.label,
                                        columns=self.feature_list,
                                        sparse=True)

        #### Reindex and transporse to get columns of get dummy #########
        nodes_features = nodes_features.T.reindex(self.feature_list).T.fillna(0)
        nodes_features = nodes_features.values
        zero_features_for_padding_at_the_end = np.zeros((1, nodes_features.shape[1]), dtype=float)
        nodes_features = np.r_[nodes_features, zero_features_for_padding_at_the_end]

        i_list = np.reshape(self.neighbor_matrix, [-1])
        the_place_of_zero_features_for_padding = i_list.max() + 1
        i_list = np.where(i_list < 0, the_place_of_zero_features_for_padding, i_list)
        ret = nodes_features[i_list]

        ret = np.reshape(ret, (self.num_graphs, self.width * self.receptive_field, self.num_features))
        finalshape = ([self.num_graphs, self.width, self.receptive_field, self.num_features])
        ret = np.reshape(ret, finalshape)

        return ret



    def graphs_to_patchy_tensor(self,labelling_procedure = 'bc'):
        if self.relabel_performed == True:
            self.update_file_path(labelling_procedure,GRAPH_RELABEL_NAME)
        else:
            self.update_file_path(labelling_procedure)

        if self.check_if_tensor_exists(self.file_path_load):
            print('{} tensor exists, loading it from Dropbox'.format(self.dataset_name))
            print('Loading path: {}'.format(self.file_path_load))
            return np.load(self.file_path_load)
        else:
            print('{} graph tensor non exisiting at : {}'.format(self.dataset_name,self.file_path_load))
            print('Create dictionary of graphs')

            self.create_adj_coomatrix_by_graphId()
            print('Labeling:')

            #### ------>>>CHECK HERE

            #self.canonical_labeling()
            self.labelling(labelling_procedure)

            #self.df_node_label = pd.concat([self.df_node_label, pd.Series(self.cano_label, dtype=int, name='cano_label')], axis=1)
            print('Getting the Neighboors ')
            self.get_neighbor_matrix()
            print('Neighboors to Tensor')
            result_tensor = self.neighbor_to_tensor()
            self.patchy_tensor = result_tensor
            np.save(self.file_path_save,result_tensor)

            print('{} graph tensor created at: {}'.format(self.dataset_name,self.file_path_load))
            return result_tensor


    def get_average_num_nodes(self):

        if not hasattr(self,'adj_dict_by_graphId'):
            self.adj_dict_by_graphId = self.create_adj_dict_by_graphId()
        self.nodes_per_graph = []
        for graph_key in self.graph_ids:
            #adj_dict = self.adj_dict_by_graphId[graph_key]
            #self.nodes_per_graph.append(len(np.unique(adj_dict.values)))
            num_nodes = len(self.df_node_label[self.df_node_label['graph_ind']==graph_key].node.tolist())
            self.nodes_per_graph.append(num_nodes)
        self.avg_nodes_per_graph = np.mean(self.nodes_per_graph)
        return self.avg_nodes_per_graph




    def graph_to_patchy_tensor_bc(self,labeling_procedure_name='betweeness', use_node_deg=False, dummy_value=-1 ):  # X is a list of Graph objects

        if self.check_if_tensor_exists(self.file_path_load_bc):
            print('{} tensor exists, loading it from Dropbox'.format(self.dataset_name))
            print('Loading path: {}'.format(self.file_path_load_bc))
            return np.load(self.file_path_load_bc)
        else:
            start = time()
            n = len(self.nx_graphs)
            train = []
            print('Creating receptive fields: ')
            for i in self.graph_ids:
                rfMaker = ReceptiveFieldMaker(self.nx_graphs[i], w=self.width, k=self.k, s=self.s,
                                              labeling_procedure_name=labeling_procedure_name,
                                              use_node_deg=use_node_deg,
                                              one_hot=self.num_features,
                                              dummy_value=dummy_value,
                                              attr_name=self.attr_name,
                                              label_name=self.label_name)

                forcnn = rfMaker.make_()
                self.times_process_details['neigh_assembly'].append(np.sum(rfMaker.all_times['neigh_assembly']))
                self.times_process_details['normalized_subgraph'].append(np.sum(rfMaker.all_times['normalized_subgraph']))
                self.times_process_details['canonicalizes'].append(np.sum(rfMaker.all_times['canonicalizes']))
                self.times_process_details['compute_subgraph_ranking'].append(
                    np.sum(rfMaker.all_times['compute_subgraph_ranking']))
                self.times_process_details['labeling_procedure'].append(np.sum(rfMaker.all_times['labeling_procedure']))
                self.times_process_details['first_labeling_procedure'].append(
                    np.sum(rfMaker.all_times['first_labeling_procedure']))

                train.append(np.array(forcnn).flatten().reshape(self.width, self.k, self.num_features))
                # Progress Bar
                progress_bar(i, n)
            X_preprocessed = np.array(train)
            end = time()
            print('Time preprocess data in s', end - start)

            np.save(self.file_path_save_bc, X_preprocessed)

        return X_preprocessed

    def print_nodes(self,nx_graph):
        for i in nx_graph.nodes:
            print(i, nx_graph.node[i])

    def print_dict(self,dictio):
        for i in dictio.items():
            print(i)


    def gen_iter_attributes_dict(self,df_nodes):
        #    for i in df_nodes.label.unique():
        #        yield(i,df_nodes[df_nodes['label']==i].node.tolist())
        for node, value in df_nodes.values:
            yield (node, {self.attr_name: value})


    def get_nx_graph(self,df_nodes, df_edges):
        graph = nx.Graph()
        graph.add_nodes_from(self.gen_iter_attributes_dict(df_nodes))
        graph.add_edges_from(df_edges)
        return graph

    def create_nx_graphs(self):
        nx_graphs = {}
        for i in self.graph_ids:
            df_nodes = self.node_label_by_graphId[i].loc[:,['node','label']]
            df_edges = self.adj_dict_by_graphId[i].loc[:,['from','to']].values
            nx_graph = self.get_nx_graph(df_nodes,df_edges)
            nx_graphs[i]=nx_graph
        return nx_graphs













