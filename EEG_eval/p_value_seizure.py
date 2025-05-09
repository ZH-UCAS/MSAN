# -*- coding: utf-8 -*-


import numpy as np
from scipy.special import comb, perm

class P_value:
    def __init__(self, tw0, tw, N, n, pw):
        self.tw0=tw0
        self.tw=tw
        self.N=N
        self.n=n
        self.pw=pw
        
    def fb(self, k, n, p):
        return comb(n, k)*np.power(p, k)*np.power((1-p), (n-k))

    def Fb(self, k, n, p):
        sum_value = 0
        for j in range(0,k+1):
            sum_value += self.fb(j, n, p)
        return sum_value
    
    def calculate_p(self):  
        numda=(-1)/self.tw*np.log(1-self.pw)
        # print("numda:", numda)
        # print("self.tw:", self.tw)
        # print("self.tw0:", self.tw0)
        # print("self.N:", self.N)
        # print("self.n:", self.n)
        # print("self.pw:", self.pw)
        Snc=1-np.exp(-numda*self.tw+(1-np.exp(-numda*self.tw0)))
        # print("self.N:", self.N)
        # print("Snc:", Snc)
        # print("self.n:", self.n)
        kf=int(np.floor(2*self.N*Snc-self.n))
        p=1 - self.Fb(self.n-1, self.N, Snc) + self.Fb(kf, self.N, Snc)
        return p
    
