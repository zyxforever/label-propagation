
class Agent:

    def __init__(self):
        self.cfg=config()
        self.dataset=Dataset(self.cfg).load_dataset()
        
    def config(self):
        parser = argparse.ArgumentParser(description='uncertainty')
        parser.add_argument('--dataset_path',default='/home/zyx/datasets')
        parser.add_argument('--model',default='cnn', choices=['cnn', 'dropout', 'scissors'])
        parser.add_argument('--data_set',default='mnist')
        parser.add_argument('--train_batch_size',default=128)
        parser.add_argument('--epoch',default=50)
        parser.add_argument('--learning_rate',default=1e-2)
        parser.add_argument('--cuda',default=True)
        return parser.parse_args()

    def run(self):
        
if __name__=='__main__':
    agent=Agent()
    agent.run()