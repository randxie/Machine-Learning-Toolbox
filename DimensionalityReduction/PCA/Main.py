'''
Created on Nov 7, 2014

@author: Yuyang
'''
import numpy as np
from PCA import PCAalgor
import matplotlib.pyplot as plt

data=np.loadtxt("data2.txt",delimiter=',')
#plt.plot(data[:,0],data[:,1],'g*')
#plt.show()

PCAalgorithm=PCAalgor(data)
PCAalgorithm.Normalization()
PCAalgorithm.CalCoVar()
PCA_eigen=PCAalgorithm.PCA_SVD()

plt.plot(PCAalgorithm.data[:,0],PCAalgorithm.data[:,1],'ro')
print(PCA_eigen)
plt.plot(np.array([0,PCA_eigen[0,0]]),np.array([0,PCA_eigen[1,0]]),'g')
plt.plot(np.array([0,PCA_eigen[1,0]]),np.array([0,PCA_eigen[1,1]]),'g')
plt.show()
