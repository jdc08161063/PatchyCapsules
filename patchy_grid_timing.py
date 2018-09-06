

# coding: utf-8

# In[1]:


import json
#import pydevd
#pydevd.settrace('localhost', port=49309, stdoutToServer=True, stderrToServer=True)
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from six.moves import xrange
#from __pynauty__ import graph
#import pynauty.graph
import pynauty
import time
import tensorflow as tf
import networkx as nx
import numpy as np
import pynauty as nauty
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import os


from PatchyTools import Dataset, utils


# ### Load data

# In[2]:



'''
#Node id is made to start from 0 due to nauty package requirement, even if it starts from 1 in the original
#Graph id is starting from 1
'''

mutag = Dataset.Dropbox('MUTAG')
df_edge_label = mutag.get_edge_label()
df_graph_ind = mutag.get_graph_ind()
df_adj = mutag.get_adj()
df_node_label = mutag.get_node_label()
df_node_label = pd.concat([df_node_label, df_graph_ind.graph_label], axis=1)
del df_graph_ind



# ### Functions

# In[3]:



def get_subset_adj(df_adj, df_node_label,graph_label_num):
    df_glabel = df_node_label[df_node_label.graph_label == graph_label_num ]
    index_of_glabel = (df_adj['to'].isin(df_glabel.node) & df_adj['from'].isin(df_glabel.node))
    return df_adj[index_of_glabel]

def get_smallest_node_id_from_adj(df_adj):
    return min(df_adj['to'].min(), df_adj['from'].min())


def create_adj_dict_by_graphId(df_adj, df_node_label):
    '''
    input: df_node_label
    return: {1: {0:[0,2,5]}} = {graphId: {nodeId:[node,node,node]}}
    '''
    adj_dict_by_graphId ={}
    unique_graph_labels = df_node_label.graph_label.unique()
    for l in unique_graph_labels:
        df_subset_adj = get_subset_adj(df_adj, df_node_label, graph_label_num=l)
        smallest_node_id = get_smallest_node_id_from_adj(df_subset_adj)
        df_subset_adj -= smallest_node_id
        adj_dict_by_graphId[l] = df_subset_adj
    return adj_dict_by_graphId


def canonical_labeling(adj_dict_by_graphId, df_node_label, df_adj):
    all_canonical_labels =[]
    unique_graph_labels = df_node_label.graph_label.unique()
    for l in unique_graph_labels:
        df_subset_adj = adj_dict_by_graphId[l]
        df_subset_nodes = df_node_label[df_node_label.graph_label==l]
        temp_graph_dict = utils.dfadj_to_dict(df_subset_adj)
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
    unique_graph_labels = df_node_label.graph_label.unique()
    for l in unique_graph_labels:
        df_subset_adj = adj_dict_by_graphId[l]
        df_subset_node_label = df_node_label[df_node_label.graph_label == l]
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

    unique_graph_labels = df_node_label.graph_label.unique()
    for l in unique_graph_labels:
        adjacency = adj_coomatrix_by_graphId[l]
        graph = nx.from_numpy_matrix(adjacency.todense())

        # Create the neighbors with -1 for neighbor assemble.
        #After this, if the RECEPTIVE_FIELD_SIZE_K exceeds the number of WIDTH_W, then fill them with -1
        neighborhoods = np.zeros((WIDTH_W, RECEPTIVE_FIELD_SIZE_K), dtype=np.int32)
        neighborhoods.fill(-1)

        df_sequence = df_node_label[df_node_label.graph_label == l]
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
                neighborhoods[i][j] = df_shortest['node'].values[j]

        neighborhoods_dict[l]= neighborhoods.copy()
    return neighborhoods_dict




# ### Arguments

# In[4]:



now = time.time()


# # Main (Timing starts here)

# In[5]:




#NUM_NODES
RECEPTIVE_FIELD_SIZE_K = 20 # Receptive Field Size K
WIDTH_W = 8 #threshold based on canonical label

adj_dict_by_graphId = create_adj_dict_by_graphId(df_adj, df_node_label)
cano_label = canonical_labeling(adj_dict_by_graphId, df_node_label, df_adj)
df_node_label = pd.concat([df_node_label, pd.Series(cano_label, dtype=int, name='cano_label')],  axis=1)

#cert_list = [i for i in (nauty.certificate(nauty_graph))]
# '''canonical_labeling = [df_node_label.label.values[i] for i in canonical_labeling]'''


# ###### Show the frequency of labels to make threshold

# In[6]:


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

# In[7]:

feature_list = df_node_label['label'].unique()
num_features = len(feature_list)
adj_coomatrix_by_graphId = create_adj_coomatrix_by_graphId(adj_dict_by_graphId, df_node_label)





# ### Things about tensorflow constraction
# In[8]:



# In[10]:

def combine_features(graph_id, neighborhoods_graph, WIDTH_W, RECEPTIVE_FIELD_SIZE_K):
    """
    neighborhoods[graphId]: This represents the matrix of (nodes x neighbor).
    nodes: This represents the matrix of (nodes x features).
    """

    neighborhoods = tf.constant(neighborhoods_graph[graph_id], dtype=tf.int32)

    sparse_df = pd.get_dummies(df_node_label.loc[df_node_label.graph_label==graph_id].label,
                               columns=feature_list,
                               sparse=True )

    #### Reindex and transporse to get columns of get dummy #########
    sparse_df = sparse_df.T.reindex(feature_list).T.fillna(0)
    nodes = tf.constant(sparse_df.values, dtype=tf.int32 )

    with tf.Session() as sess:
        print ('shape of neighborhoods', neighborhoods.shape)
        data = tf.reshape(neighborhoods, [-1])
        i = tf.maximum(data, 0)
        i_list = i.eval()
        #print(i_list)


        #tf.gather_nd should be used here.


        for ind, i in enumerate(i_list):
            if ind ==0:
                positive = tf.strided_slice(nodes, [i], [i+1], [1])
                #positive = tf.constant(temp)
            else:
                temp = tf.strided_slice(nodes, [i], [i+1], [1])
                positive = tf.concat([positive,temp],axis=0)

        negative = tf.zeros([positive.shape[0], positive.shape[1]], dtype= tf.int32)

        #print ('shape', data.shape, negative.shape, positive.shape)
        #print ( data, negative, positive)

        # padding with 0 here because -1 indicates there is no neighbor nodes.
        ret = tf.where(data < 0, negative, positive)
        #print('padding done')


        ret = tf.reshape(ret,
                         [WIDTH_W,
                          RECEPTIVE_FIELD_SIZE_K,
                          num_features])
        #print (key, 'ret.shape: ',ret.shape)
        ret = ret.eval()

    return ret

#stop stp stp



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)




def main(WIDTH_W, RECEPTIVE_FIELD_SIZE_K):
    global now
    neighborhoods_graph = make_neighbor(adj_coomatrix_by_graphId, df_node_label, WIDTH_W=WIDTH_W, RECEPTIVE_FIELD_SIZE_K=RECEPTIVE_FIELD_SIZE_K)
    print ('after neibor assemble: ', time.time()-now)
    for key in adj_dict_by_graphId.keys():
        result_tensor = combine_features(key, neighborhoods_graph,WIDTH_W, RECEPTIVE_FIELD_SIZE_K)
    return  result_tensor



#list_WIDTH_W=[12,15,18,21,24]
#list_RECEPTIVE_FIELD_SIZE_K=[2,3,4,5,10]
list_WIDTH_W=[15]
list_RECEPTIVE_FIELD_SIZE_K=[2]
secondsret = pd.DataFrame(np.zeros((len(list_WIDTH_W),len(list_RECEPTIVE_FIELD_SIZE_K)),dtype=float))

for w_ind, w in enumerate(list_WIDTH_W):
    for k_ind, k in enumerate(list_RECEPTIVE_FIELD_SIZE_K):
        print (k, w, 'has been started')
        now = time.time()
        result_patcy = main(w,k)
        #print ('result_patcy', result_patcy)
        seconds = time.time() - now
        print ('after neibor assemble: ', time.time() - now)

        secondsret.iloc[w_ind, k_ind] = seconds

        #secondsret.to_csv('mutag_seconds_result.csv')
        #save_object(result_patcy, 'patchy_np_array_mutag_w{}k{}.pkl'.format(w,k))


