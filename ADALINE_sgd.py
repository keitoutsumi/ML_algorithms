import numpy as np

class ADALINEBGD(object):

    def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        self.random_state=random_state

    def fit(self,X,y):
        self._initialize_weights(X.shape[1])
        self.cost_=[]

    
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,-1)

    def initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized=True