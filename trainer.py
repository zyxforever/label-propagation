from models.gcn import GCN
from models.gat import GAT
import os
import glob
import torch 
import models 
import logging 
import argparse 
import torch.optim as optim
import torch.nn.functional as F

class Trainer:

    def __init__(self,cfg,data_set):
        self.cfg=cfg 
        self.data_set=data_set
        
        adj, features, labels, idx_train, idx_val, idx_test=self.data_set
        self.adj=adj
        self.features=features
        self.labels=labels
        self.idx_train=idx_train
        self.idx_val=idx_val
        self.idx_test =idx_test
        
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
        self.logger=logging.getLogger(__name__)
        if self.cfg.model=='gcn':
            self.model=GCN(nfeat=features.shape[1],
                nhid=self.cfg.hidden,
                nclass=labels.max().item() + 1,
                dropout=self.cfg.dropout)
        elif self.cfg.model=='gat':
            self.model = GAT(
                nfeat=features.shape[1],
                nhid=self.cfg.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=self.cfg.dropout, 
                nheads=self.cfg.nb_heads, 
                alpha=self.cfg.alpha)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)


    def train(self):
        if self.cfg.model=='gcn':
            for i in range(self.cfg.epochs):
                self.trains() 
            self.test()
        elif self.cfg.model=='gat':
            loss_values = []
            bad_counter = 0
            best = self.cfg.epochs + 1
            best_epoch = 0
            for epoch in range(self.cfg.epochs):
                loss_values.append(self.trains())

                torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))
                if loss_values[-1] < best:
                    best = loss_values[-1]
                    best_epoch = epoch
                    bad_counter = 0
                else:
                    bad_counter += 1

                if bad_counter == self.cfg.patience:
                    break

                files = glob.glob('*.pkl')
                for file in files:
                    epoch_nb = int(file.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(file)

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(file)
            self.model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
            self.test()




    
    def trains(self):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])            
        acc_train = self.evaluate(output[self.idx_train], self.labels[self.idx_train])            
        loss_train.backward()
        self.optimizer.step()
        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val= self.evaluate(output[self.idx_val], self.labels[self.idx_val])            
        self.logger.info("loss_train {:.4f}, acc_train{:.4f}, loss_val:{:.4f}, acc_val:{:.4f}".format(loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item() ))
        if self.cfg.model=='gat':
            return loss_val.data.item() 
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
        self.logger.info("loss {:.4f}, accuracy{:.4f}".format(loss_test.item(),acc_test.item() ))