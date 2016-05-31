'''
Created on Nov 8, 2014

@author: Yuyang
'''
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from Sample import SVM
from cvxopt import matrix
from cvxopt import solvers
from SVMalgor import SVMalgor
def ShowData(data):
    for i in range(0,data.shape[0]):
        if data[i,2]==1:
            plt.plot(data[i,0],data[i,1],'go')
        else:
            plt.plot(data[i,0],data[i,1],'r^')
  
    
data=np.loadtxt("data2.txt",delimiter=',')
'''
SVMsolver=SVMalgor(data)
SVMsolver.QP_solver()
weight=SVMsolver.ReturnW()
print weight
ShowData(data)
plt.plot(np.array([0,1]),np.array([0,-weight[0,0]/weight[0,1]]),'yo')
plt.show()
'''
x=np.random.rand(4,1)
print x
print x>0.5