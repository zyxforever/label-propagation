import models 
import logging 
import argparse 

from tools import Tools
from tqdm import trange,tqdm 
from trainer import Trainer
from dataset import Dataset
from sklearn.metrics import accuracy_score as accuracy
from algorithms.label_propagation import LPA

class Agent:
    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        parser.add_argument('--baselines',default=['lgc'])
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--model',default='gat')
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--dataset_path',default='data/cora')
        parser.add_argument('--data_set',default='coil')
        parser.add_argument('--label_rate',default=0.05,type=float)
        parser.add_argument('--num_knn',default=10,type=int)
        parser.add_argument('--distance',default='knn')
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--epochs',default=10000)
        parser.add_argument('--lr',default=0.005)
        parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
        parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        parser.add_argument('--patience', type=int, default=100, help='Patience')
        parser.add_argument('--cuda',default=False)
        return parser.parse_args()

    def __init__(self):
        self.cfg=self.config()
        self.logger=Tools.get_logger(__name__)
        self.logger.info("Label Rate:{},DataSet:{},num_knn:{}"\
                .format(self.cfg.label_rate,self.cfg.data_set,self.cfg.num_knn))
        self.data_loader=Dataset(self.cfg)
        self.data_set=self.data_loader.load_dataset()
        if self.cfg.data_set in ['mnist10k','letters','coil']:
            self.adj,self.feature,self.labels=self.data_set

            '''
            self.idx_confuse,self.idx_confuse_more,self.idx_pure,self.idx_pure_confuse=\
                Dataset.entropy_adj(adj,labels[:int(self.cfg.label_rate*len(labels))],labels)
            self.logger.info("idx_confuse:%s,idx_confuse_more:%s,idx_pure:%s,idx_pure_confuse:%s"
            %(self.idx_confuse.shape[0],self.idx_confuse_more.shape[0],self.idx_pure.shape[0],self.idx_pure_confuse.shape[0]))
            '''
        else:
            adj, features, labels, idx_train, idx_val, idx_test=self.data_set
    
    def evaluate(self,output):
        acc=accuracy(output,self.labels)
        '''
        acc_confuse=accuracy(output[self.idx_confuse],self.labels[self.idx_confuse]) \
            if self.idx_confuse.shape[0]>0 else 0
        acc_confuse_more=accuracy(output[self.idx_confuse_more],self.labels[self.idx_confuse_more]) \
            if self.idx_confuse_more.shape[0]>0 else 0
        acc_pure=accuracy(output[self.idx_pure],self.labels[self.idx_pure]) \
            if self.idx_pure.shape[0]>0 else 0 
        acc_pure_confuse=accuracy(output[self.idx_pure_confuse],self.labels[self.idx_pure_confuse]) \
            if self.idx_pure_confuse.shape[0]>0 else 0 
        self.logger.info("Acc_Confuse:%.4f,Acc_confuse_more:%.4f,Acc_pure:%.4f,Acc_pure_confuse:%.4f"
                    %(acc_confuse,acc_confuse_more,acc_pure,acc_pure_confuse))
        '''
        self.logger.info("Acc:%s"%acc)
        self.logger.info("---------------------------")
    def run(self):
        for baseline in self.cfg.baselines:
            if baseline=='lgc':
                output=LPA.lgc(self.adj,self.labels,len(self.labels),int(len(self.labels)*self.cfg.label_rate),self.logger)
                self.evaluate(output)
            elif baseline=='lgc_new':
                self.adj=self.data_loader.weight_matrix()
                output=LPA.lgc_new(self.adj,self.labels,len(self.labels),int(len(self.labels)*self.cfg.label_rate),self.logger)
            elif baseline=='lnp':
                pass 
            elif baseline=='gcn':
                self.cfg.model='gcn'
                trainer=Trainer(self.cfg,self.data_set)
                trainer.train()
            elif baseline=='gat':
                self.cfg.model='gat'
                trainer=Trainer(self.cfg,self.data_set)
                trainer.train()

if __name__=='__main__':
    agent=Agent()
    agent.run()