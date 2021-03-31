import math
import torch 
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm 
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics.cluster import adjusted_rand_score as ari 
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.neighbors import kneighbors_graph

def load_usps_mat(shuffle=True):
    vec=scipy.io.loadmat('data_PenDigits_bai.mat')
    N=vec['fea'].shape[0]
    idx_rand=np.random.choice(N,size=N,replace=False)
    X=vec['fea'][idx_rand] if shuffle else vec['fea']
    Y=vec['gt'][idx_rand] if shuffle else vec['gt']
    Y=np.squeeze(Y)
    return X,Y
alpha = 0.9
sigma = 0.2
#X = np.loadtxt("moon_data.txt")
#Y = np.loadtxt("class.txt")

print("load data...")
#X, Y = make_moons(n, shuffle=True, noise=0.1, random_state=None)
#X,Y =load_usps()
X,Y=load_usps_mat()
n=X.shape[0]
n_labeled_list=[int(n*0.05),int(n*0.1),int(n*0.15),int(n*0.2),int(n*0.25),int(n*0.30)]

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_laplician(X):
    dm = cdist(X, X, 'sqeuclidean')
    rbf = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
    vfunc = np.vectorize(rbf)
    W = vfunc(dm, sigma)
    np.fill_diagonal(W, 0)
    sum_lines = np.sum(W,axis=1)
    D = np.diag(sum_lines)
    print("compute sys...")
    D = fractional_matrix_power(D, -0.5)
    S = np.dot(np.dot(D,W), D)
    return S 

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def compute_knn(X):
    adj = kneighbors_graph(X, 10,mode='connectivity', include_self=True)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    #adj = sparse_mx_to_torch_sparse_tensor(adj) 
    return adj.toarray()
S=compute_knn(X)
for n_labeled in n_labeled_list:
    class_set=np.unique(Y)
    class_num=len(class_set)
    sigma=np.std(X)
    Y_input = np.concatenate(((Y[:n_labeled,None] == np.arange(class_num)).astype(float), np.zeros((n-n_labeled,class_num))))
    print("compute distance...")
    #S=compute_laplician(X)
    #S=S.numpy()
    n_iter = 400
    print("compute iter...")
    # Use a new S 

    F = np.dot(S, Y_input)*alpha + (1-alpha)*Y_input
    for t in tqdm(range(n_iter)):
        F = np.dot(S, F)*alpha + (1-alpha)*Y_input
    Y_result = np.zeros_like(F)
    Y_result[np.arange(len(F)), F.argmax(1)] = 1
        #Y_v = [1 if x == 0 else 0 for x in Y_result[0:,0]]
    #y_v=[x for x in Y_result[0:,0]]
    y_v=[np.argmax(one_hot) for one_hot in Y_result]
    print("ACC NMI ARI")
    print("{:0.4f}  {:0.4f}  {:0.4f}".format(acc(y_v,Y),nmi(y_v,Y),ari(y_v,Y)))
    #plt.scatter(X[0:,0], X[0:,1], color=color)
    #plt.savefig("iter_n.png")
    #plt.show()