

def compute_knn(X):
    adj = kneighbors_graph(X, 10,mode='connectivity', include_self=True)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj.toarray()

    #adj = sparse_mx_to_to
def lgc(X,Y):
    S=compute_knn(X)
    