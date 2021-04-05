import sys
import torch
import pickle as pkl
import numpy as np 
import networkx as nx
import scipy.io
import scipy.sparse as sp

from sklearn.neighbors import kneighbors_graph

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Dataset:
    def __init__(self,cfg):
        self.cfg=cfg  
    def load_dataset(self):
        if self.cfg.data_set=='mnist10k':
            return self.load_data_mat()
        return self.load_data()
    def load_data(self):    
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            file="%s/ind.%s.%s"%(self.cfg.dataset_path,self.cfg.data_set, names[i])
            with open(file, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_file="%s/ind.%s.test.index"%(self.cfg.dataset_path,self.cfg.data_set)
        test_idx_reorder = parse_index_file(test_file)
        test_idx_range = np.sort(test_idx_reorder)
        if self.cfg.data_set == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = normalize(features)
        adj = preprocess_adj(adj)

        features = torch.FloatTensor(np.array(features.todense())).float()
        labels = torch.LongTensor(labels)
        labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        return adj, features, labels, idx_train, idx_val, idx_test

    def load_data_mat(self,shuffle=True):
        # 读取特征矩阵
        fea = scipy.io.loadmat(self.cfg.dataset_path)
        # 获取特征矩阵 行 列大小
        N = fea['data'].shape[0]
        M = fea['data'].shape[1]
        # 随机抽样
        idx_rand = np.random.choice(N, size=N, replace=False)
        X = fea['data'][idx_rand] if shuffle else fea['data']
        Y = fea['labels'][idx_rand] if shuffle else fea['labels']
        Y = np.squeeze(Y)
        adj=self.compute_knn(X)
        return adj,X, Y, N, M

    @staticmethod
    def entropy(Y):
        num_classes=np.bincount(Y)
        probs=num_classes/len(Y)
        return -np.sum(probs*np.log(probs))/np.log(len(np.unique(Y)))
    def compute_knn(self,X):
        adj = kneighbors_graph(X, 10,mode='connectivity', include_self=True)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        #adj = sparse_mx_to_torch_sparse_tensor(adj) 
        return adj.toarray()