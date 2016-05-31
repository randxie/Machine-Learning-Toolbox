'''
Created on Nov 8, 2014

@author: Yuyang
'''
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from scipy import optimize

class SVMalgor():
    def __init__(self,data):
        self.data=data
        self.Q=-1*np.ones((data.shape[0],1))
        self.P=self.InnerP()
        self.h=np.zeros((data.shape[0],1))
        self.G=-1*np.eye(data.shape[0])
        self.b=0
        self.A=matrix(data[:,data.shape[1]-1], (1,data.shape[0]))
        
    def InnerP(self):
        data=self.data
        y=data[:,data.shape[1]-1]
        x=data[:,0:data.shape[1]-1]
        xMtx=np.zeros((data.shape[0],data.shape[0]))
        for i in range(0,data.shape[0]):
            for j in range(0,data.shape[0]):
                xMtx[i,j]=np.dot(x[i,:],x[j,:])
        return matrix(np.outer(y,y)*xMtx)
        
    def QP_solver(self):
        P=matrix(self.P)
        Q=matrix(self.Q)
        G=matrix(self.G)
        h=matrix(self.h)
        A=self.A
        b=matrix(0.0)
        self.sol = solvers.qp(P,Q,G,h,A,b)
    
    def ReturnW(self):
        data=self.data
        y=data[:,data.shape[1]-1]
        x=data[:,0:data.shape[1]-1]
        w=np.zeros((data.shape[1]-1,1))
        alpha=np.ravel(self.sol['x'])
        for i in range(0,data.shape[0]):
            w=w+alpha[i]*y[i]*x[i,0:data.shape[1]-1]
        self.w=w
        return self.w
'''
QP solver example
# Define QP parameters (with NumPy)
P = matrix(np.diag([1,0]),tc='d')
q = matrix(np.array([3,4]),tc='d')
G = matrix(np.array([[-1,0],[0,-1],[-1,-3],[2,5],[3,4]]),tc='d')
h = matrix(np.array([0,0,-15,100,80]),tc='d')
# Construct the QP, invoke solver
sol = solvers.qp(P,q,G,h)
# Extract optimal value and solution
sol['x'] # [7.13e-07, 5.00e+00]
sol['primal objective'] # 20.0000061731
'''