"Patchy San"

from networkx import nx
from networkx import convert_node_labels_to_integers
try:
    import pynauty as nauty
    from pynauty.graph import canonical_labeling
except:
    print('Pynauty is not installed in this environment')
from scipy.sparse import coo_matrix
from DropboxLoader import DropboxLoader

import copy

import numpy as np
import time

import utils




class ReceptiveFieldMaker(object):
    def __init__(self, nx_graph, w, s=1, k=10, labeling_procedure_name='betweeness', use_node_deg=False, one_hot=False,
                 dummy_value=-1, attr_name='attr_name', label_name='labelling'):
        self.nx_graph = nx_graph
        self.use_node_deg = use_node_deg
        if self.use_node_deg:
            node_degree_dict = dict(self.nx_graph.degree())
            normalized_node_degree_dict = {k: v / len(self.nx_graph.nodes()) for k, v in node_degree_dict.items()}
            nx.set_node_attributes(self.nx_graph, normalized_node_degree_dict, self.attr_name)
        self.all_times = {}
        self.all_times['neigh_assembly'] = []
        self.all_times['normalized_subgraph'] = []
        self.all_times['canonicalizes'] = []
        self.all_times['compute_subgraph_ranking'] = []
        self.all_times['labeling_procedure'] = []
        self.all_times['first_labeling_procedure'] = []
        self.w = w
        self.s = s
        self.k = k
        self.dummy_value = dummy_value
        self.exists_dummies = False
        self.one_hot = one_hot
        self.labeling_procedure_name = labeling_procedure_name
        self.attr_name = attr_name
        self.label_name = label_name


        if self.labeling_procedure_name == 'approx_betweeness':
            st = time.time()
            self.dict_first_labeling = self.betweenness_centrality_labeling(self.nx_graph, approx=int(
                len(self.nx_graph.nodes()) / 5) + 1)
            self.labeling_procedure_name = 'betweeness'
            end = time.time()
            self.all_times['first_labeling_procedure'].append(end - st)
        elif self.labeling_procedure_name == 'betweeness':
            st = time.time()
            self.dict_first_labeling = self.betweenness_centrality_labeling(self.nx_graph)
            end = time.time()
            self.all_times['first_labeling_procedure'].append(end - st)
        else:
            st = time.time()
            self.dict_first_labeling = self.labeling_procedure(self.nx_graph)
            end = time.time()
            self.all_times['first_labeling_procedure'].append(end - st)

        self.original_labeled_graph = self.dict_first_labeling['labeled_graph']

    def make_(self):
        "Result on one (w,k,length_attri) list (usually (w,k,1)) for 1D CNN "
        forcnn = []
        self.all_subgraph = []
        f = self.select_node_sequence()
        for graph in f:
            frelabel = nx.relabel_nodes(graph,
                                        nx.get_node_attributes(graph, self.label_name))  # rename the nodes wrt the labeling
            self.all_subgraph.append(frelabel)
            if self.one_hot > 0:
                forcnn.append([utils.indices_to_one_hot(x[1], self.one_hot) for x in
                               sorted(nx.get_node_attributes(frelabel, self.attr_name).items(), key=lambda x: x[0])])
            else:
                forcnn.append(
                    [x[1] for x in sorted(nx.get_node_attributes(frelabel, self.attr_name).items(), key=lambda x: x[0])])
        return forcnn

    def labeling_procedure(self, graph):
        st = time.time()
        if self.labeling_procedure_name == 'betweeness':
            a = self.betweenness_centrality_labeling(graph)
        end = time.time()
        self.all_times['labeling_procedure'].append(end - st)
        return a

    def betweenness_centrality_labeling(self, graph, approx=None):
        result = {}
        labeled_graph = nx.Graph(graph)
        if approx is None:
            centrality = list(nx.betweenness_centrality(graph).items())
        else:
            centrality = list(nx.betweenness_centrality(graph, k=approx).items())
        sorted_centrality = sorted(centrality, key=lambda n: n[1], reverse=True)
        dict_ = {}
        label = 0
        for t in sorted_centrality:
            dict_[t[0]] = label
            label += 1
        nx.set_node_attributes(labeled_graph, dict_, self.label_name)
        ordered_nodes = list(zip(*sorted_centrality))[0]

        result['labeled_graph'] = labeled_graph
        result['sorted_centrality'] = sorted_centrality
        result['ordered_nodes'] = ordered_nodes
        return result

    def wl_normalization(self, graph):

        result = {}

        labeled_graph = nx.Graph(graph)

        relabel_dict_ = {}
        graph_node_list = list(graph.nodes())
        for i in range(len(graph_node_list)):
            relabel_dict_[graph_node_list[i]] = i
            i += 1

        inv_relabel_dict_ = {v: k for k, v in relabel_dict_.items()}

        graph_relabel = nx.relabel_nodes(graph, relabel_dict_)

        label_lookup = {}
        label_counter = 0

        l_aux = list(nx.get_node_attributes(graph_relabel, self.attr_name).values())
        labels = np.zeros(len(l_aux), dtype=np.int32)
        adjency_list = list([list(x[1].keys()) for x in
                             graph_relabel.adjacency()])  # adjency list à l'ancienne comme version 1.0 de networkx

        for j in range(len(l_aux)):
            if not (l_aux[j] in label_lookup):
                label_lookup[l_aux[j]] = label_counter
                labels[j] = label_counter
                label_counter += 1
            else:
                labels[j] = label_lookup[l_aux[j]]
            # labels are associated to a natural number
            # starting with 0.

        new_labels = copy.deepcopy(labels)

        # create an empty lookup table
        label_lookup = {}
        label_counter = 0

        for v in range(len(adjency_list)):
            # form a multiset label of the node v of the i'th graph
            # and convert it to a string

            long_label = np.concatenate((np.array([labels[v]]), np.sort(labels[adjency_list[v]])))
            long_label_string = str(long_label)
            # if the multiset label has not yet occurred, add it to the
            # lookup table and assign a number to it
            if not (long_label_string in label_lookup):
                label_lookup[long_label_string] = label_counter
                new_labels[v] = label_counter
                label_counter += 1
            else:
                new_labels[v] = label_lookup[long_label_string]
        # fill the column for i'th graph in phi
        labels = copy.deepcopy(new_labels)

        dict_ = {inv_relabel_dict_[i]: labels[i] for i in range(len(labels))}

        nx.set_node_attributes(labeled_graph, dict_, self.label_name)

        result['labeled_graph'] = labeled_graph
        result['ordered_nodes'] = [x[0] for x in sorted(dict_.items(), key=lambda x: x[1])]

        return result

    def select_node_sequence(self):
        Vsort = self.dict_first_labeling['ordered_nodes']
        f = []
        i = 0
        j = 1
        while j <= self.w:
            if i < len(Vsort):
                f.append(self.receptiveField(Vsort[i]))
            else:
                f.append(self.zeroReceptiveField())
            i += self.s
            j += 1

        return f

    def zeroReceptiveField(self):
        graph = nx.star_graph(self.k - 1)  # random graph peu importe sa tete
        nx.set_node_attributes(graph, self.dummy_value, self.attr_name)
        nx.set_node_attributes(graph, {k: k for k, v in dict(graph.nodes()).items()}, self.label_name)

        return graph

    def receptiveField(self, vertex):
        st = time.time()
        subgraph = self.neighborhood_assembly(vertex)
        ed = time.time()
        self.all_times['neigh_assembly'].append(ed - st)
        normalized_subgraph = self.normalize_graph(subgraph, vertex)
        ed2 = time.time()
        self.all_times['normalized_subgraph'].append(ed2 - ed)

        return normalized_subgraph

    def neighborhood_assembly(self, vertex):
        "Output a set of neighbours of the vertex"
        N = {vertex}
        L = {vertex}
        while len(N) < self.k and len(L) > 0:
            tmp = set()
            for v in L:
                tmp = tmp.union(set(self.nx_graph.neighbors(v)))
            L = tmp - N
            N = N.union(L)
        return self.nx_graph.subgraph(list(N))

    def rank_label_wrt_dict(self, subgraph, label_dict, dict_to_respect):

        all_distinc_labels = list(set(label_dict.values()))
        new_ordered_dict = label_dict

        latest_biggest_label = 0

        for label in all_distinc_labels:

            nodes_with_this_label = [x for x, y in subgraph.nodes(data=True) if y[self.label_name] == label]

            if len(nodes_with_this_label) >= 2:

                inside_ordering = sorted(nodes_with_this_label, key=dict_to_respect.get)
                inside_order_dict = dict(zip(inside_ordering, range(len(inside_ordering))))

                for k, v in inside_order_dict.items():
                    new_ordered_dict[k] = latest_biggest_label + 1 + inside_order_dict[k]

                latest_biggest_label = latest_biggest_label + len(nodes_with_this_label)

            else:
                new_ordered_dict[nodes_with_this_label[0]] = latest_biggest_label + 1
                latest_biggest_label = latest_biggest_label + 1

        return new_ordered_dict

    def compute_subgraph_ranking(self, subgraph, vertex, original_order_to_respect):

        st = time.time()

        labeled_graph = nx.Graph(subgraph)
        ordered_subgraph_from_centrality = self.labeling_to_root(subgraph, vertex)

        all_labels_in_subgraph_dict = nx.get_node_attributes(ordered_subgraph_from_centrality, self.label_name)

        new_ordered_dict = self.rank_label_wrt_dict(ordered_subgraph_from_centrality, all_labels_in_subgraph_dict,
                                                    original_order_to_respect)

        nx.set_node_attributes(labeled_graph, new_ordered_dict, self.label_name)
        ed = time.time()
        self.all_times['compute_subgraph_ranking'].append(ed - st)
        return labeled_graph

    def canonicalizes(self, subgraph):

        st = time.time()

        # wl_subgraph_normalized=self.wl_normalization(subgraph)['labeled_graph']
        # g_relabel=convert_node_labels_to_integers(wl_subgraph_normalized)

        g_relabel = convert_node_labels_to_integers(subgraph)
        labeled_graph = nx.Graph(g_relabel)

        nauty_graph = nauty.Graph(len(g_relabel.nodes()), directed=False)
        nauty_graph.set_adjacency_dict({n: list(nbrdict) for n, nbrdict in g_relabel.adjacency()})

        labels_dict = nx.get_node_attributes(g_relabel, self.label_name)
        canonical_labeling_dict = {k: canonical_labeling(nauty_graph)[k] for k in range(len(g_relabel.nodes()))}

        new_ordered_dict = self.rank_label_wrt_dict(g_relabel, labels_dict, canonical_labeling_dict)

        nx.set_node_attributes(labeled_graph, new_ordered_dict, self.label_name)

        ed = time.time()
        self.all_times['canonicalizes'].append(ed - st)

        return labeled_graph

    def normalize_graph(self, subgraph, vertex):

        "U set of vertices. Return le receptive field du vertex (un graph normalisé)"
        ranked_subgraph_by_labeling_procedure = self.labeling_procedure(subgraph)['labeled_graph']
        original_order_to_respect = nx.get_node_attributes(ranked_subgraph_by_labeling_procedure, self.label_name)
        subgraph_U = self.compute_subgraph_ranking(subgraph, vertex,
                                                   original_order_to_respect)  # ordonne les noeuds w.r.t labeling procedure

        if len(subgraph_U.nodes()) > self.k:

            d = dict(nx.get_node_attributes(subgraph_U, self.label_name))
            k_first_nodes = sorted(d, key=d.get)[0:self.k]
            subgraph_N = subgraph_U.subgraph(k_first_nodes)

            ranked_subgraph_by_labeling_procedure = self.labeling_procedure(subgraph)['labeled_graph']
            original_order_to_respect = nx.get_node_attributes(ranked_subgraph_by_labeling_procedure, self.label_name)
            subgraph_ranked_N = self.compute_subgraph_ranking(subgraph_N, vertex, original_order_to_respect)

        elif len(subgraph_U.nodes()) < self.k:
            subgraph_ranked_N = self.add_dummy_nodes_at_the_end(subgraph_U)
        else:
            subgraph_ranked_N = subgraph_U

        return self.canonicalizes(subgraph_ranked_N)

    def add_dummy_nodes_at_the_end(self, nx_graph):  # why 0 ??
        self.exists_dummies = True
        g = nx.Graph(nx_graph)
        keys = [k for k, v in dict(nx_graph.nodes()).items()]
        labels = [v for k, v in dict(nx.get_node_attributes(nx_graph, self.label_name)).items()]
        j = 1
        nodes_to_add = []
        while len(g.nodes()) < self.k:
            #g.add_node(max(keys) + j, attr_name=self.dummy_value, labeling=max(labels) + j)
            nodes_to_add.append((max(keys) + j, {self.attr_name: self.dummy_value, self.label_name: max(labels) + j}))
            j += 1
        g.add_nodes_from(nodes_to_add)
        return g

    def labeling_to_root(self, graph, vertex):
        labeled_graph = nx.Graph(graph)
        source_path_lengths = nx.single_source_dijkstra_path_length(graph, vertex)
        nx.set_node_attributes(labeled_graph, source_path_lengths, self.label_name)

        return labeled_graph




