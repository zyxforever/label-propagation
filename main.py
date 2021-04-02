
import torch 
import models 
import logging 
import argparse 
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange,tqdm 
from models.gcn import GCN 
from algorithms.lp import lgc
from dataset import Dataset

logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.cfg=self.config()
        self.data_set=Dataset(self.cfg).load_dataset()
        adj, features, labels, idx_train, idx_val, idx_test=self.data_set
        self.model = GCN(nfeat=features.shape[1],
                nhid=self.cfg.hidden,
                nclass=labels.max().item() + 1,
                dropout=self.cfg.dropout)
        if self.cfg.cuda:
            self.model.cuda()
            self.features = features.cuda()
            self.adj = adj.cuda()
            self.labels = labels.cuda()
            self.idx_train = idx_train.cuda()
            self.idx_val = idx_val.cuda()
            self.idx_test = idx_test.cuda()
        # Model and optimizer
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        
        parser.add_argument('--baselines',default='lgc', choices=['lgc', 'dropout', 'scissors'])
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--hidden',type=int,default=16)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--dataset_path',default='/home/zyx/datasets/cora')
        parser.add_argument('--data_set',default='cora')
        parser.add_argument('--label_rate',default=0.05)
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--epochs',default=200)
        parser.add_argument('--lr',default=0.01)
        parser.add_argument('--cuda',default=True)
        return parser.parse_args()
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = self.evaluate(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()
        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val= self.evaluate(output[self.idx_val], self.labels[self.idx_val])
        logger.info("loss_train {:.4f}, acc_train{:.4f}, loss_val:{:.4f}, acc_val:{:.4f}".format(loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item() ))
    def evaluate(self,output,labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = self.evaluate(output[self.idx_test], self.labels[self.idx_test])
        logger.info("loss {:.4f}, accuracy{:.4f}".format(loss_test.item(),acc_test.item() ))

    def run(self):
        for epoch in trange(self.cfg.epochs):
           self.train()
        self.test()

if __name__=='__main__':
    agent=Agent()
    agent.run()