import os 
class Tools:
    @staticmethod
    def cmd():
        for i in range(100):
            os.system('python main.py --label_rate 0.01')
if __name__=='__main__':
    #Tools.cmd()
    '''
    x=[0.96889,0.9831,0.9848,0.9952,]
    y=[]
    '''
    Tools.cmd()