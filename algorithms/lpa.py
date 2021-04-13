import numpy as np 

class LPA:
    @staticmethod
    def lgc(S,labels,n,n_labeled,logger,open=False):
        n_iter=400
        alpha=0.99
        sigma=0.6
        class_set=np.unique(labels)
        class_num=len(class_set)
        Y_input = np.concatenate(((labels[:n_labeled,None] == np.arange(class_num)).astype(float), np.zeros((n-n_labeled,class_num))))
        # 添加标签平滑项
        if open:
            idx_row=Y_input[:n_labeled].argmax(1)
            for i,row in enumerate(idx_row):
                Y_input[i]=sigma/(class_num-1)
                Y_input[i][row]=1-sigma
            Y_input[n_labeled:]= 1/class_num
        F = np.dot(S, Y_input)*alpha + (1-alpha)*Y_input
        for t in range(n_iter):
            F = np.dot(S, F)*alpha + (1-alpha)*Y_input
        #entropy=-np.sum(F*np.log(F))/np.log(class_num)
        #logger.info("Entropy of F: %s"%entropy)
        output = np.zeros_like(F)
        output[np.arange(len(F)), F.argmax(1)] = 1
        predict=[np.argmax(one_hot) for one_hot in output]
        predict=np.array(predict)
        return predict
    @staticmethod  
    def lgc_new(S,labels,n,n_labeled,logger,open=False):
        for i in range(10):
            pass 
if __name__=='__main__':
    LPA.lgc()