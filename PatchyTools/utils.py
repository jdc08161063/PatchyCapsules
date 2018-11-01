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




def indices_to_one_hot(number, nb_classes, label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""

    if number == label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]



def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()