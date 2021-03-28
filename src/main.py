
import logging 
import argparse 

from algorithms.lp import lgc
from dataset import Dataset

logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)

class Agent:

    def __init__(self):
        self.cfg=self.config()
        self.data_set=Dataset(self.cfg).load_dataset()
        
    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        
        parser.add_argument('--model',default='cnn', choices=['cnn', 'dropout', 'scissors'])
        parser.add_argument('--dataset_path',default='/home/zyx/datasets/MNIST10k.mat')
        parser.add_argument('--data_set',default='mnist')
        parser.add_argument('--label_rate',default=0.05)
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--epoch',default=50)
        parser.add_argument('--learning_rate',default=1e-2)
        parser.add_argument('--cuda',default=True)
        return parser.parse_args()

    def run(self):
        X,Y,N,M=self.data_set
        lgc(X,Y)
if __name__=='__main__':
    agent=Agent()
    agent.run()