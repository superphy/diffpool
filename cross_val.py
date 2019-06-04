import networkx as nx
import numpy as np
import torch

import pickle
import random
import typing

from graph_sampler import GraphSampler


def prepare_val_data(graphs: typing.List[str], args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    # print('Number of graphs: ', len(graphs))
    # print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    # print('Max, avg, std of graph size: ',
    #         max([G.number_of_nodes() for G in graphs]), ', '
    #         "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
    #         "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    print("Creating GraphSampler instances....")
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    print("Done creating GraphSampler instances")
    print("Trying to create DataLoader instances with batch_size {} and "
          "num_workers {}".format(args.batch_size, args.num_workers))
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)
    print("Done creating DataLoader instances for training")

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)
    print("Done creating GraphSampler instances...")

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

