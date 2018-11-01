#!/usr/bin/env python3
# coding: utf-8
"""
Implementation of Capsule Networks:
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from time import time

import argparse
from collections import defaultdict

from utils import plot_log, save_results_to_csv
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

sys.path.append('./PatchyTools/')
from PatchyConverter import PatchyConverter
from DropboxLoader import DropboxLoader
from CapsuleParameters import CapsuleParameters
from CapsuleParameters import CapsuleTrainingParameters

# from ConvNetPatchy import AccuracyHistory

DIR_PATH = os.environ['GAMMA_DATA_ROOT']
GRAPH_RELABEL_NAME = '_relabelled'
RESULTS_PATH = os.path.join(DIR_PATH, 'Results/CapsuleSans/CNN_Caps_comparison_bc.csv')



if __name__ == "__main__":

    # Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name_of the dataset', default='MUTAG')
    parser.add_argument('-k', help='receptive field for patchy', default=10)
    parser.add_argument('-st', help='stride field for patchy', default=1)
    parser.add_argument('-r', dest='relabelling', help='reshuffling takes place', action='store_true')
    parser.add_argument('-nr', dest='relabelling', help='no reshuffling takes place', action='store_false')
    parser.set_defaults(relabelling=True)

    # parser.add_argument('-sampling_ratio', help='ratio to sample on', default=0.2)

    # Parsing arguments:
    args = parser.parse_args()

    # Arguments:
    dataset_name = args.n
    # width = int(args.w)
    receptive_field = int(args.k)
    relabelling = args.relabelling
    stride = int(args.st)

    # Converting Graphs into Matrices:
    graph_converter = PatchyConverter(dataset_name, receptive_field,stride)

    print('Graph imported')
    if relabelling:
        print('Relabelling:')
        graph_converter.relabel_graphs()

    graph_tensor = graph_converter.graphs_to_patchy_tensor()
    avg_nodes_per_graph = graph_converter.avg_nodes_per_graph
