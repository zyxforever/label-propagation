from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph


def compute_knn(X):
    adj = kneighbors_graph(X, 10,mode='connectivity', include_self=True)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj.toarray()

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
    
def lgc(X,Y):
    S=compute_laplician(X)
    
