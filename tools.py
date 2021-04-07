import os 
import logging 
from logging import handlers
class Tools:
    @staticmethod
    def cmd():
        for i in range(100):
            os.system('python main.py --label_rate 0.01')

    def get_logger(name):
        logging.basicConfig(level = logging.INFO, filename='new.log',filemode='a',format = '%(asctime)s-%(name)s -%(levelname)s-%(message)s')
        logger=logging.getLogger(name)
        th = handlers.TimedRotatingFileHandler(filename='logs/log.log',when='D',backupCount=2,encoding='utf-8')
        format_str = logging.Formatter('%(asctime)s-%(name)s -%(levelname)s-%(message)s')#设置日志格式
        th.setFormatter(format_str) 
        logger.addHandler(th)
        return logger 
if __name__=='__main__':
    #Tools.cmd()
    '''
    x=[0.96889,0.9831,0.9848,0.9952,]
    y=[]
    '''
    Tools.cmd()