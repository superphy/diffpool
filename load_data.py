import networkx as nx
import numpy as np
import scipy as sc
import os
import re

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname)
    print("Looking for graph files in: {}".format(prefix))
    files = [
        os.path.join(prefix, f)
        for f in os.listdir(prefix)
        if f.endswith('.g')
    ]
    print("Found {} .g files".format(len(files)))
    return files

