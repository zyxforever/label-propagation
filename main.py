import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import random
import scipy.io
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.linalg import fractional_matrix_power
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi,adjusted_rand_score as ari
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from sklearn.metrics import accuracy_score as acc
import sklearn.preprocessing
from scipy.special import entr

def parse_args():
    parser = argparse.ArgumentParser(description="dataset and method")
    parser.add_argument('--dataset', nargs='+',required=True)
    parser.add_argument('--method',nargs='+',required=True)
    parser.add_argument('--label_rate',default=10,help='默认10')
    #实验运行次数
    parser.add_argument('--times',default=10,help='默认每个算法运行10次，求平均')
    parser.add_argument('--type',default='bl',choices=['bl','imbl'],help='平衡还是非平衡')
    parser.add_argument('--n_iter',type=int,default=4)
    args = parser.parse_args()

    return args
def extract_data(X, Y, l):
    N_X = np.c_[X, Y]
    ll = np.where(N_X[:, X.shape[1]] == l)[0]
    random.shuffle(ll)
    return ll
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def SIS_matrix(X,Y):
    X = X / np.max(X)
    l1 = np.unique(Y).tolist()
    # k = int(len(l1) * 1.5)
    k=20
    dist = pdist(X, 'euclidean')  # 获得距离矩阵
    dist = squareform(dist)  # 转化为方阵
    m = dist.shape[0]
    k_idx = np.zeros([m, k])
    for i in range(m):
        topk = np.argsort(dist[i])[1:k + 1]  # 从1开始，是因为最小那个距离是它本身, 返回最小的k个的索引
        k_idx[i] = k_idx[i] + topk
    k_idx=k_idx.astype(np.int32)
    w = np.zeros([m, m])
    for i in range(m):
        Q_x = X[k_idx[i]]
        xi = X[i]
        xi = np.tile(xi, (k, 1))
        C = np.dot((xi - Q_x), (xi - Q_x).T)
        C = C + np.eye(k) * (1e-3) * np.trace(C)
        C_inv = np.linalg.pinv(C)
        tol=np.sum(C_inv)
        if tol ==0:
            tol+=1e-6
        l=np.sum(C_inv, axis=0) / tol
        l=np.maximum(l,0)
        t = np.sum(l)
        if t ==0:
            t+=1e-6
        l=l/t
        w[i, k_idx[i]] = l
    return w
def compute_knn(X,k):
    adj = kneighbors_graph(X, k,mode='connectivity', include_self=True)
    # print(adj)
    # print(type(adj))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj.toarray()
def DLP_matrix(X,k):
    sigma = 0.1
    X = X / 255
    X = X /np.max(X)
    dm = cdist(X, X, 'euclidean')
    rbf = lambda x, sigma: math.exp((-x) / (2 * (math.pow(sigma, 2))))
    vfunc = np.vectorize(rbf)
    W = vfunc(dm, sigma)
    P=np.zeros_like(W)
    sort=np.argsort(-W)
    sort=sort[:,0:k]
    for i in range(W.shape[0]):
        P[i, sort[i]]=W[i, sort[i]]
    P=normalize(P)
    W=W/np.sum(W)
    return W,P
def GFHF_matrix(X):
    sigma = 0.1
    X = X / np.max(X)
    dm = cdist(X, X, 'euclidean')
    rbf = lambda x, sigma: math.exp((-x) / (2 * (math.pow(sigma, 2))))
    vfunc = np.vectorize(rbf)
    W = vfunc(dm, sigma)
    np.fill_diagonal(W, 0)
    sum_lines = np.sum(W, axis=0)
    D = np.diag(sum_lines)
    D = fractional_matrix_power(D, -1)
    W=np.dot(D,W)
    return W
def weight_matrix(X):
    sigma = 0.1
    X=X/np.max(X)
    dm = cdist(X, X, 'euclidean')
    rbf = lambda x, sigma: math.exp((-x) / (2 * (math.pow(sigma, 2))))
    vfunc = np.vectorize(rbf)
    W = vfunc(dm, sigma)
    np.fill_diagonal(W, 0)
    sum_lines = np.sum(W, axis=1)
    D = np.diag(sum_lines)
    D = fractional_matrix_power(D, -0.5)
    S = np.dot(np.dot(D, W), D)
    return S
def progatation1(S,Y,n_labeled,n,n_iter):
    alpha = 0.99
    labels_list = len(np.unique(Y))
    # 将标签向量lx1通过比较张成对应列为1的矩阵 lxc大小
    # print(np.array((Y[:n_labeled, None] == np.arange(labels_list))).astype(int))
    # 拼接无标签数据扩充至nxc大小
    Y0=np.array((Y[:n_labeled, None] == np.arange(labels_list))).astype(int)

    Y_input = np.concatenate((Y0, np.zeros((n - n_labeled, labels_list))))

    F = np.dot(S, Y_input) * alpha + (1 - alpha) * Y_input
    for t in range(n_iter):

        F = np.dot(S, F) * alpha + (1 - alpha) * Y_input
    return F
def GFHF_progatation1(S, Y, n_labeled, n, n_iter):
    labels_list = len(np.unique(Y))
    Y0 = (Y[:n_labeled, None] == np.arange(labels_list)).astype(float)
    Y_input = np.concatenate((Y0, np.zeros((n - n_labeled, labels_list))))
    F = np.dot(S, Y_input)
    for t in range(n_iter):
        F = np.dot(S, F)
        F = sklearn.preprocessing.normalize(F, axis=1, norm='max')

    return F
def DLP_progatation1(S, P, Y, n_labeled, n, n_iter):
    alpha = 0.05
    lamda = 0.1
    p = P
    labels_list = len(np.unique(Y))
    # 将标签向量lx1通过比较张成对应列为1的矩阵 lxc大小
    # print((Y[:n_labeled, None] == np.arange(labels_list)).astype(float))
    # 拼接无标签数据扩充至nxc大小
    Y0 = (Y[:n_labeled, None] == np.arange(labels_list)).astype(float)
    Y = np.concatenate((Y0, np.zeros((n - n_labeled, labels_list))))
    Y = np.dot(p, Y)
    P = np.dot(np.dot(S, (p + alpha * (np.dot(Y, Y.T)))), S.T) + lamda * np.eye(n)
    for t in range(n_iter):
        # print(t)
        Y = np.dot(P, Y)
        # print(Y[:10])
        Y = np.r_[Y0, Y[n_labeled:, :]]

        P = np.dot(np.dot(S, (P + alpha * (np.dot(Y, Y.T)))), S.T) + lamda * np.eye(n)
    return Y
def DLP_progatation(S,P,Y_input,n,n_iter):
    alpha = 0.05
    lamda=0.1
    p=P

    Y=Y_input
    Y=np.dot(p,Y)
    P = np.dot(np.dot(S, (p + alpha * (np.dot(Y, Y.T)))), S.T) + lamda * np.eye(n)
    for t in range(n_iter):
        Y=np.dot(P,Y)
        P=np.dot(np.dot(S, (P+alpha*(np.dot(Y,Y.T)))),S.T)+lamda*np.eye(n)
    return Y
def GFHF_progatation(S,Y_input,n_iter):
    F = np.dot(S, Y_input)
    for t in range(n_iter):
        F = np.dot(S, F)
        F = sklearn.preprocessing.normalize(F,axis=1,norm='max')
    return F
def progatation(S,Y_input,n_iter):
    alpha = 0.99
    F = np.dot(S, Y_input) * alpha + (1 - alpha) * Y_input
    for t in range(n_iter):
        F = np.dot(S, F) * alpha + (1 - alpha) * Y_input
    return F
def predict(F,Y,n_labeled):
    Y_result = np.zeros(F.shape[0] - n_labeled)
    for i in range(F.shape[0] - n_labeled):
        Y_result[i] = np.argmax(F[i + n_labeled])
    Y_p = np.r_[Y[:n_labeled], Y_result]
    return format(acc(Y_p, Y),'.4f'),format(nmi(Y_p, Y),'.4f'),format(ari(Y_p, Y),'.4f')
def class_balance_data(shuffle=True,dataset='data_PenDigits_bai',label_rate=10):

    #读取特征矩阵
    data = scipy.io.loadmat('{}.mat'.format(dataset))


    fea = data['X']
    gt = data['y']
    n_labeled = int(fea.shape[0]*(label_rate/100))
    gt=np.squeeze(gt).tolist()
    g_t=list(np.unique(gt))
    c=len(g_t)
    p=int(n_labeled/c)
    random.shuffle(g_t)

    # 获取特征矩阵 行 列大小
    N = fea.shape[0]
    M = fea.shape[1]
    l=[]
    for i in g_t:
        l1=extract_data(fea,gt,i)
        l1=l1[:p]
        l.extend(l1)

    X1=fea[l]
    gt=np.array(gt)
    Y1=gt[l]
    # print('数据集：', Counter((Y1).flatten()))

    fea=fea[np.setdiff1d(np.arange(N),l)]
    gt=gt[np.setdiff1d(np.arange(N),l)]
    N=fea.shape[0]
    #随机抽样
    idx_rand = np.random.choice(N, size=N, replace=False)
    X2=fea[idx_rand] if shuffle else fea
    Y2=gt[idx_rand] if shuffle else gt
    X=np.r_[X1,X2]
    Y=np.r_[Y1,Y2]
    Y=np.squeeze(Y)
    N = X.shape[0]
    M = X.shape[1]
    # print('数据集：', Counter((Y).flatten()))
    return X,Y,N,M
def extract(Y,Y_in,c,num):
    t=[]
    l=(Y==c).nonzero()[0]
    random.shuffle(l)
    for i in l:
        if i not in Y_in:
            t.append(i)
        if len(t)==num:
            break
    return t
def score(F,labeled):
    mean_s= np.mean(np.max(F[labeled],axis=1))
    a=F[labeled]
    s=a.sum(axis=1,keepdims=True)
    idx1=np.setdiff1d(np.arange(len(labeled)),(s==0).nonzero()[0])
    b=a[idx1]
    b=b/b.sum(axis=1,keepdims=True)
    ea1 = np.sum(entr(b).sum(axis=1))/np.log(2)
    ea1 = ea1/len(idx1)

    m = F.sum(axis=1, keepdims=True)
    idx2 = np.setdiff1d(np.arange(F.shape[0]), (m == 0).nonzero()[0])
    c= F[idx2]
    c = c / c.sum(axis=1, keepdims=True)
    ea2 = np.sum(entr(c).sum(axis=1)) / np.log(2)
    ea2 = ea2 / len(idx2)
    return mean_s,ea1,ea2
def k_nn(X,k,l):
    dist = pdist(X, 'euclidean')  # 获得距离矩阵
    dist = squareform(dist)  # 转化为方阵
    sort = np.argsort(dist)
    sort = sort[l, 1:k+1].ravel()
    return sort
def search_F(F,i):
    idx = F.argmax(1)
    val_max = F[idx == i][:, i]
    idx_sort = val_max.argsort()[::-1]
    l=(idx == i).nonzero()[0][idx_sort]
    return l
def judge_F(N_l,L,l,num):
    add=[]
    m=l[:2*num]
    for i in N_l:
        if i in m and i not in L:
            add.append(i)
        if len(add)==num:
            break
    if len(add)<num:
        for i in l:
            if i not in N_l and i not in L:
                add.append(i)
            if len(add)==num:
                break

    return add
def initial_Y_input(L,Y,n,m):
    labels=np.unique(Y)
    labels=[ int(i) for i in labels]
    labeled = []
    Y_input = np.zeros((n, len(labels)))
    for i in labels:
        l=(Y==i).nonzero()[0]

        if len(l)>m:
            q=random.sample(list(l),m)

            labeled.extend(q)
            Y_input[q,i]=1
        else:
            labeled.extend(l)
            Y_input[l,i]=1
    return Y_input,labeled
def search_Y(Y,c,labeled):
    l = (Y.argmax(1)==c).nonzero()[0]
    L=[]
    for i in l:
        if i in labeled:
            L.append(i)
    return L
def class_imblance_data(shuffle=True,dataset='data_PenDigits_bai',label_rate=10):
    # 读取特征矩阵
    data = scipy.io.loadmat('{}.mat'.format(dataset))

    fea = data['fea']
    gt = data['gt']
    n_labeled = int(fea.shape[0] * (label_rate / 100))
    g_t = np.unique(gt).tolist()
    gt = np.squeeze(gt).tolist()
    p=[]
    p1=[2,3,6,9,16,26,43,72,120,200]
    p2=[1, 3, 7, 10]
    p3=[1,4,10]
    if len(g_t)==10:

        p.extend(p1)
    elif len(g_t)==4:

        p.extend(p2)
    elif len(g_t)==3:

        p.extend(p3)


    p = [i / sum(p) for i in p]



    # 获取特征矩阵 行 列大小
    N = fea.shape[0]
    M = fea.shape[1]
    random.shuffle(g_t)
    # g_t=[3,1,4,2,5,0]
    l = []
    ll = []

    for i in range(len(g_t)):
        l1 = extract_data(fea, gt, g_t[i])
        l1 = list(l1[:int(p[i] * n_labeled)])
        ll.append(l1)
        l.extend(l1)
    X1 = fea[l]
    gt = np.array(gt)
    Y1 = gt[l]
    # print('数据集：', Counter((Y1).flatten()))

    fea = fea[np.setdiff1d(np.arange(N), l)]
    gt = gt[np.setdiff1d(np.arange(N), l)]
    N = fea.shape[0]
    idx_rand = np.random.choice(N , size=N , replace=False)
    X2 = fea[idx_rand] if shuffle else fea
    Y2 = gt[idx_rand] if shuffle else gt
    X = np.r_[X1, X2]
    Y = np.r_[Y1, Y2]
    Y = np.squeeze(Y)
    N = X.shape[0]
    M = X.shape[1]
    return X, Y, N, M,g_t,ll,p
def balance(times,dataset,method,label_rate,n_iter):
    result_bl = []
    for i in range(times):
        A = []
        X, Y, n, m = class_balance_data(dataset=dataset,label_rate=label_rate)
        n_labeled=int(n*(label_rate/100))
        # print(X.shape)
        if method == 'lgc':
            S = weight_matrix(X)
            F = progatation1(S, Y, n_labeled, n, n_iter)
        elif method == 'lnp':
            S = compute_knn(X, 15)
            F = progatation1(S, Y, n_labeled, n, n_iter)
        elif method == 'sis':
            S = SIS_matrix(X, Y)
            F = progatation1(S, Y, n_labeled, n, n_iter)
        elif method == 'gfhf':
            S = GFHF_matrix(X)
            F = GFHF_progatation1(S, Y, n_labeled, n, n_iter)
        elif method == 'dlp':
            S, P = DLP_matrix(X, 10)
            F = DLP_progatation1(P, S, Y, n_labeled, n, n_iter)
        else:
            print('no this method,select from lgc, lnp, sis, dlp, gfhf')

        a, b, c = predict(F, Y,n_labeled)
        a = float(a)
        b = float(b)
        c = float(c)
        l1 = [a, b, c]
        A.append(l1)
        result_bl.append(A)
    result1 = np.array(result_bl)
    m_bl = np.mean(result1, axis=0)
    std = np.std(result_bl, axis=0)
    print('mean-balance,dataset:{},method:{}'.format(dataset,method),'ACC,NMI,ARI', m_bl,'标准差',std)
def imbalance(times,dataset,method,label_rate,n_iter):
    result_acc = []
    result_nmi = []
    result_ari = []
    result_score = []
    result_com = []
    for i in range(times):
        A = []
        B = []
        C = []
        E1 = []
        E2 = []
        F_ = []
        X, Y, n, m, g_t, L,p = class_imblance_data(dataset=dataset,label_rate=label_rate)
        n_labeled=int(n*(label_rate/100))
        g_t = [int(i) for i in g_t]
        num = [int(i * n_labeled) for i in p]
        balance_num = []
        for i in range(len(num) - 1):
            balance_num.append(int(num[i + 1] - num[i]))
        balance_num1 = []

        k = int((max(num) - min(num)) / (len(num) - 1))
        for i in range(len(num) - 2):
            balance_num1.append(k * (i + 1))
        balance_num1.append(max(num))

        Y_input, labeled = initial_Y_input(L, Y[:n_labeled], n, num[0])
        # print(np.sum(Y_input))
        # print('Y_input：', Counter((Y[labeled]).flatten()))
        F=np.zeros((n,m))

        if method == 'lgc':
            S = weight_matrix(X)
            F = progatation(S, Y_input, n_iter)
        elif method == 'lnp':
            S = compute_knn(X, 15)
            F = progatation(S, Y_input, n_iter)
        elif method == 'sis':
            S = SIS_matrix(X, Y)
            F = progatation(S, Y_input, n_iter)
        elif method == 'gfhf':
            S = GFHF_matrix(X)
            F = GFHF_progatation(S, Y_input, n_iter)
        elif method == 'dlp':
            S, P = DLP_matrix(X, 10)
            F = DLP_progatation(P, S, Y_input, n, n_iter)
        else:
            print('no this method,select from lgc, lnp, sis, dlp, gfhf')

        F_.append(F.tolist())
        a, b, c = predict(F, Y,n_labeled)
        d, e, f = score(F, labeled)
        E1.append(float(e))
        E2.append(float(f))
        A.append(float(a))
        B.append(float(b))
        C.append(float(c))

        for i in range(1, len(g_t)):
            # for i in range(1,3):
            Y_l = Y[:n_labeled]
            Y_L = labeled
            c = g_t[:i]

            # F
            for j in c:
                # print(balance_num[i-1])
                L = search_Y(Y_input, j, labeled)
                # print('已经标记',L)
                N_l = k_nn(X, 3, L)
                # print('近邻',N_l)
                # print('近邻个数',len(N_l))
                l = search_F(F, j)
                # print('F中分高的',l)
                ll = judge_F(N_l, L, l, balance_num[i - 1])
                # print(pre(list([j] * (len(ll))), list(Y[ll])))
                # print('选出的下标',ll)
                # print('选出的下标',len(ll))
                Y_input[ll, j] = 1
                labeled.extend(ll)
            # 补齐有标记的数据
            t_c = g_t[i:]
            for j in t_c:
                ll = extract(Y_l, labeled, j, balance_num[i - 1])
                # print('选出的下标', ll)
                # print('选出的下标', len(ll))
                Y_input[ll, j] = 1
                labeled.extend(ll)
            # print(np.sum(Y_input))
            # print('带标记', len(labeled))
            Y_result = Y[labeled]
            # print('Y_input：', Counter((Y_result).flatten()))

            # 继续传播
            if method == 'lgc':
                F = progatation(S, Y_input, n_iter)
            elif method == 'lnp':
                F = progatation(S, Y_input, n_iter)
            elif method == 'sis':
                F = progatation(S, Y_input, n_iter)
            elif method == 'gfhf':
                F = GFHF_progatation(S, Y_input, n_iter)
            elif method == 'dlp':
                F = DLP_progatation(P, S, Y_input, n, n_iter)
            else:
                print('no this method,select from lgc, lnp, sis, dlp, gfhf，第二次')

            F_.append(F.tolist())

            a, b, c = predict(F, Y,n_labeled)
            d, e, f = score(F, labeled)

            E1.append(float(e))
            E2.append(float(f))
            A.append(float(a))
            B.append(float(b))
            C.append(float(c))

        F_ = np.array(F_)

        item=[]

        E1 = [math.exp(-math.pow(i, 2) / (2 * (math.pow((sum(E2) / len(E2)), 2)))) for i in E2]
        E1 = [i / sum(E1) for i in E1]
        F_com1 = F_[0] * E1[0]
        for i in range(1, len(E1)):
            F_com1 += F_[i] * E1[i]
        # print('融合1 高斯核')
        a2, b2, c2 = predict(F_com1, Y,n_labeled)
        # print(a2, b2, c2)
        item.append(float(a2))
        item.append(float(b2))
        item.append(float(c2))

        result_acc.append(A)
        result_nmi.append(B)
        result_ari.append(C)
        result_score.append(S)
        result_com.append(item)
    result1 = np.array(result_acc)
    result2 = np.array(result_nmi)
    result3 = np.array(result_ari)
    result4 = np.array(result_com)
    m_acc = np.mean(result1, axis=0)
    m_nmi = np.mean(result2, axis=0)
    m_ari = np.mean(result3, axis=0)
    m_fuse = np.mean(result4, axis=0)

    std = np.std(result4, axis=0)

    print('mean-imbalance,dataset:{},method:{}'.format(dataset, method), 'ACC,NMI,ARI', m_fuse, '标准差', std)
if __name__ == '__main__':
    args = parse_args()
    if args.type=='bl':
        for i in args.dataset:
            for j in args.method:
                balance(args.times, i, j, args.label_rate, args.n_iter)
    elif args.type=='imbl':
        for i in args.dataset:
            for j in args.method:
                imbalance(args.times, i, j, args.label_rate, args.n_iter)


