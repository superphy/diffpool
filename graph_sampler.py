import networkx as nx
import numpy as np
import torch
import torch.utils.data

import typing

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''

    def parse(self, G: nx.Graph) -> typing.Tuple[
        np.ndarray,
        typing.Optional[np.ndarray],
        typing.Optional[np.ndarray],
        typing.Any]:
        adj, feature, assign_feat, label = None, None, None, None

        adj = np.array(nx.to_numpy_matrix(G))
        if self.normalize:
            sqrt_deg = np.diag(
                1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
            adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
        # self.len_all.append(G.number_of_nodes())
        label = G.graph['label']
        # feat matrix: max_num_nodes x feat_dim
        if self.features == 'default':
            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            for i, u in enumerate(G.nodes()):
                f[i, :] = G.node[u]['feat']
            feature = f
        elif self.features == 'id':
            feature = np.identity(self.max_num_nodes)
        elif self.features == 'deg-num':
            degs = np.sum(np.array(adj), 1)
            degs = np.expand_dims(
                np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                axis=1)
            feature = degs
        elif self.features == 'deg':
            self.max_deg = 10
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs > self.max_deg] = self.max_deg
            feat = np.zeros((len(degs), self.max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            feat = np.pad(feat, (
            (0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                          'constant', constant_values=0)

            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            for i, u in enumerate(G.nodes()):
                f[i, :] = G.node[u]['feat']

            feat = np.concatenate((feat, f), axis=1)

            feature = feat
        elif self.features == 'struct':
            self.max_deg = 10
            degs = np.sum(np.array(adj), 1).astype(int)
            degs[degs > 10] = 10
            feat = np.zeros((len(degs), self.max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            degs = np.pad(feat, (
            (0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                          'constant', constant_values=0)

            clusterings = np.array(list(nx.clustering(G).values()))
            clusterings = np.expand_dims(np.pad(clusterings,
                                                [0,
                                                 self.max_num_nodes - G.number_of_nodes()],
                                                'constant'),
                                         axis=1)
            g_feat = np.hstack([degs, clusterings])
            if 'feat' in G.node[0]:
                node_feats = np.array(
                    [G.node[i]['feat'] for i in range(G.number_of_nodes())])
                node_feats = np.pad(node_feats, (
                (0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                    'constant')
                g_feat = np.hstack([g_feat, node_feats])

            feature = g_feat

        if self.assign_feat == 'id':
            assign_feat = np.hstack(
                    (np.identity(self.max_num_nodes), feature))
        else:
            assign_feat = feature

        return adj, feature, assign_feat, label

    @property
    def feat_dim(self):
        if self._feat_dim is not None:
            return self._feat_dim
        else:
            _, feature, _, _ = self.parse(
                nx.read_gpickle(self.G_list[0])
            )
            self._feat_dim = feature.shape[1]
            return self._feat_dim

    @property
    def assign_feat_dim(self):
        if self._assign_feat_dim is not None:
            return self._assign_feat_dim
        else:
            _, _, assign_feat, _ = self.parse(
                nx.read_gpickle(self.G_list[0])
            )
            self._assign_feat_dim = assign_feat
            return self._assign_feat_dim

    def __init__(self, G_list, features='default', normalize=True,
                 assign_feat='default', max_num_nodes=0, feat_dim=None):
        # Class attributes
        self.features = features
        self.normalize = normalize
        self.assign_feat = assign_feat
        # Attributes that need to be pre-calculated
        self.max_num_nodes = max_num_nodes
        self._feat_dim = feat_dim
        self._assign_feat_dim = None
        # Lists
        self.G_list = G_list

        if max_num_nodes == 0:
            current_max = 0
            for G in G_list:
                n = nx.read_gpickle(G).number_of_nodes()
                if n > current_max:
                    current_max = n
            self.max_num_nodes = current_max
            print("Setting max_num_nodes to {}".format(self.max_num_nodes))
        else:
            self.max_num_nodes = max_num_nodes

    def __len__(self):
        return len(self.G_list)

    def _load(self, f: str):
        G = nx.read_gpickle(f)
        return self.parse(G)

    def __getitem__(self, idx):
        adj, feature, assign_feat, label = self._load(self.G_list[idx])
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                'feats':feature.copy(),
                'label':label,
                'num_nodes': num_nodes,
                'assign_feats':assign_feat.copy()}

