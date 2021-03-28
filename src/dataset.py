import scipy.io
import numpy as np 
class Dataset:

    def __init__(self,cfg):
        self.cfg=cfg  

    def load_dataset(self):
        return self.load_data_mat()
        
    def load_data_mat(self,shuffle=True):
        # 读取特征矩阵
        fea = scipy.io.loadmat(self.cfg.dataset_path)
        # 获取特征矩阵 行 列大小
        N = fea['data'].shape[0]
        M = fea['data'].shape[1]
        # 随机抽样
        idx_rand = np.random.choice(N, size=N, replace=False)
        X = fea['labels'][idx_rand] if shuffle else fea['fea']
        Y = fea['gt'][idx_rand] if shuffle else fea['gt']
        Y = np.squeeze(Y)
        return X, Y, N, M