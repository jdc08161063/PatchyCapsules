from collections import OrderedDict, defaultdict
import numpy as np

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




