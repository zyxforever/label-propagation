import numpy as np 
from abc import abstractmethod

class LabelPropagation:
    @abstractmethod
    def lgc(S,labels,n,n_labeled,class_num):
        n_iter=400
        alpha=0.99
        Y_input = np.concatenate(((labels[:n_labeled,None] == np.arange(class_num)).astype(float), np.zeros((n-n_labeled,class_num))))
        F = np.dot(S, Y_input)*alpha + (1-alpha)*Y_input
        for t in range(n_iter):
            F = np.dot(S, F)*alpha + (1-alpha)*Y_input
        output = np.zeros_like(F)
        output[np.arange(len(F)), F.argmax(1)] = 1
        predict=[np.argmax(one_hot) for one_hot in output]
        return predict 

if __name__=='__main__':
    LabelPropagation.lgc()