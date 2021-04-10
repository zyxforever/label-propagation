import sys
import torch
import pickle as pkl
import numpy as np 
import networkx as nx
import scipy.io
import scipy.sparse as sp
from tools import Tools
from sklearn.neighbors import kneighbors_graph
logger=Tools.get_logger(__name__)
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

class Dataset:
    def __init__(self,cfg):
        self.cfg=cfg  
    def load_dataset(self):
        if self.cfg.data_set in ['mnist10k','letters']:
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
        data_name_path={'letters':'data/MNIST10k.mat','usps':'data/USPS.mat','letters':'data/Letters.mat'}
        self.cfg.dataset_path=data_name_pat[self.cfg.data_set]
        feature_name='data'
        label_name='labels'
        if self.cfg.data_set in ['letters','usps']:
            feature_name='fea'
            label_name='gt'
            self.cfg.dataset_path='data/Letters.mat'
        
        # 读取特征矩阵
        fea = scipy.io.loadmat(self.cfg.dataset_path)
        # 获取特征矩阵 行 列大小
        N = fea[feature_name].shape[0]
        M = fea[feature_name].shape[1]
        # 随机抽样
        idx_rand = self.rand_idx(fea[label_name])
        X = fea[feature_name][idx_rand] if shuffle else fea[feature_name]
        Y = fea[label_name][idx_rand] if shuffle else fea[label_name]
        Y = np.squeeze(Y)
        if self.cfg.data_set=='letters':
            Y=Y-1
        X=X/255
        adj=self.compute_knn(X)
        return adj,X, Y, N, M

    @staticmethod
    def entropy(Y):
        num_classes=np.bincount(Y)
        probs=num_classes/len(Y)
        return -np.sum(probs*np.log(probs))/np.log(len(np.unique(Y)))
    '''
        idx_confuse:      idx of nodes have more than one labeled neighborhood
        idx_confuse_more: the true label is not in the 
        idx_idx_pure:     idx of nodes have only one labeled neighborhood
        idx_idx_confuse:  idx of the nodes whose has only one labeled neigborhood 
                            and it's label is not equal to its neighborhood's 
    '''
    @staticmethod 
    def entropy_adj(adj,Y_L,Y):
        N=len(Y_L)
        idx_confuse=[]
        idx_confuse_more=[]
        idx_pure=[]
        idx_pure_confuse=[]
        for i,label in enumerate(Y_L): 
            indices=(adj[i]>0).nonzero()[0]
            for idx in indices:
                idx_con=(adj[idx]>0).nonzero()[0]
                idx_labeled=idx_con[idx_con<N]
                num_labeled=idx_labeled.shape[0]
                #idx_labeled=((adj[idx]>0).nonzero()[0]<N).nonzero()[0]
                if num_labeled==1 and idx not in idx_pure and idx not in idx_pure_confuse:
                    if not Y[idx] in np.unique(Y[idx_labeled]):
                        idx_pure_confuse.append(idx)
                    else:
                        idx_pure.append(idx)
                        #logger.info("Pure_Confuse i:%s,idx:%s,Y_less:%s,Y:%s"%(i,idx,Y[idx_labeled],Y[idx]))
                elif num_labeled>1 and idx not in idx_confuse and idx not in idx_confuse_more:
                    if not Y[idx] in np.unique(Y[idx_labeled]):
                        idx_confuse_more.append(idx)
                    else:
                        idx_confuse.append(idx)
        idx_confuse=np.array(idx_confuse)
        idx_confuse_more=np.array(idx_confuse_more)
        idx_pure=np.array(idx_pure)
        idx_pure_confuse=np.array(idx_pure_confuse)
        return idx_confuse,idx_confuse_more,idx_pure,idx_pure_confuse

    def compute_knn(self,X):
        adj = kneighbors_graph(X, self.cfg.num_knn,mode='connectivity', include_self=True)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        #adj = sparse_mx_to_torch_sparse_tensor(adj) 
        return adj.toarray()
    
    # generate the rand_idx for data 
    def rand_idx(self,Y):
        N=len(Y)
        idx_rand=np.random.choice(N, size=N, replace=False)
        return idx_rand