import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sio as sio
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi, adjusted_rand_score as ari


def load_data_mat(shuffle=True):
    # 读取特征矩阵
    fea = scipy.io.loadmat('data_PenDigits_bai.mat')
    # 获取特征矩阵 行 列大小
    N = fea['fea'].shape[0]
    M = fea['fea'].shape[1]
    # 随机抽样
    idx_rand = np.random.choice(N, size=N, replace=False)
    X = fea['fea'][idx_rand] if shuffle else fea['fea']
    Y = fea['gt'][idx_rand] if shuffle else fea['gt']
    Y = np.squeeze(Y)
    return X, Y, N, M


# 03年
def weight_matrix(X):
    sigma = 0.1
    dm = cdist(X, X, 'euclidean')
    # print('dm=', dm)
    rbf = lambda x, sigma: math.exp((-x) / (2 * (math.pow(sigma, 2))))
    # print(rbf)
    vfunc = np.vectorize(rbf)
    # print(vfunc)
    W = vfunc(dm, sigma)
    np.fill_diagonal(W, 0)
    # print('w=',W)
    sum_lines = np.sum(W, axis=1)
    D = np.diag(sum_lines)
    # print('d=',D)
    D = fractional_matrix_power(D, -0.5)
    S = np.dot(np.dot(D, W), D)
    return S

#08年
def KNN_matrix(X, k):
    Dist = cdist(X, X, 'euclidean')
    # print(Dist)
    sortedDist = np.argsort(Dist)
    sortedDist = sortedDist[:, 1:k + 1]
    # print(sortedDist)
    if k > len(sortedDist):
        k = len(sortedDist)
    k_Dist = np.sort(Dist)[:, 1:k + 1]
    # weight=weight[:, 1:k + 1]
    # print(k_Dist)
    sum1 = np.sum(k_Dist, axis=1)
    # print(sum1)
    a = sum1[:, None] / k_Dist
    # print(a)
    sum2 = np.sum(a, axis=1)
    weight = a / sum2[:, None]
    # print(weight)
    weight_matrix = np.zeros_like(Dist)
    # print(weight_matrix.shape)
    for i in range(X.shape[0]):
        for j in range(k):
            weight_matrix[i, sortedDist[i, j]] = weight[i, j]
    return weight_matrix


def progatation(S, Y, n_labeled, n, n_iter):
    alpha = 0.99
    labels_list = len(np.unique(Y))
    # 将标签向量lx1通过比较张成对应列为1的矩阵 lxc大小
    # print((Y[:n_labeled, None] == np.arange(labels_list)).astype(float))
    # 拼接无标签数据扩充至nxc大小
    Y_input = np.concatenate(
        ((Y[:n_labeled, None] == np.arange(labels_list)).astype(float), np.zeros((n - n_labeled, labels_list))))
    F = np.dot(S, Y_input) * alpha + (1 - alpha) * Y_input
    for t in range(n_iter):
        print(t)
        F = np.dot(S, F) * alpha + (1 - alpha) * Y_input
    return F


# 预测结果
def predict_result(predict_labels, true_labels, n_labeled):
    # acc
    sum = 0
    tol = len(predict_labels)
    for i in range(tol):
        if predict_labels[i] == true_labels[n_labeled + i]:
            sum += 1
    print('predict_acc={}'.format(sum / tol))
    # nmi,ari
    true_labels = true_labels[n_labeled:]
    predict_nmi = nmi(true_labels, predict_labels)
    predict_ari = ari(true_labels, predict_labels)
    print('predict_nmi={}'.format(predict_nmi))
    print('predict_ari={}'.format(predict_ari))


def start(methond='KNN',k=10):
    n_labeled = 50
    n_iter = 40

    X, Y, n, m = load_data_mat()
    X = X / 255
    # X=X[:500]
    if methond=='KNN':
        S = KNN_matrix(X, k)
    elif methond=='GSK':
        S=weight_matrix(X)

    F = progatation(S, Y, n_labeled, n, n_iter)
    Y_result = np.zeros(n - n_labeled)
    # print(Y_result.shape)
    # print(F)
    # print(F.shape)
    for i in range(n - n_labeled):
        Y_result[i] = np.argmax(F[i + n_labeled])
    predict_result(Y_result, Y, n_labeled)


if __name__ == '__main__':
    start(methond='GSK')