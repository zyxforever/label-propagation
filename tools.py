import os 
import logging 
from logging import handlers
class Tools:
    @staticmethod
    def cmd():
        #os.system('python main.py --label_rate 0.06')
        for i in range(3):
            os.system('python main.py --label_rate 0.1 --num_knn 3')
        for i in range(3):
            os.system('python main.py --label_rate 0.1 --num_knn 5')
        for i in range(3):
            os.system('python main.py --label_rate 0.1 --num_knn 10')
        for i in range(3):
            os.system('python main.py --label_rate 0.1 --num_knn 26')
    def get_logger(name):
        format_str = logging.Formatter('%(asctime)s-%(name)s -%(levelname)s-%(message)s')#设置日志格式
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
        logger=logging.getLogger(name)
        th = handlers.TimedRotatingFileHandler(filename='logs/log.log',when='D',backupCount=10,encoding='utf-8')
        th.setFormatter(format_str) 
        th.setLevel(logging.INFO)
        logger.addHandler(th)
        return logger 
if __name__=='__main__':
    #Tools.cmd()
    '''
    x=[0.96889,0.9831,0.9848,0.9952,]
    y=[]
    '''
    Tools.cmd()