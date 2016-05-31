'''
Created on Nov 6, 2014

@author: Yuyang
'''
from KmeanNet import KmeanNet
import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt("data2.txt",delimiter=',')
data=np.transpose(data)


#Define network structure
net=KmeanNet(5,2,5)
for i in range(0,20):
    net.Update(data)
plt.plot(data[0,:],data[1,:],'ro')
Centroid=net.ShowCentroid()
plt.plot(Centroid[0,:],Centroid[1,:],'go')
plt.show()
print(Centroid)


#Use for data generation
'''
a=np.random.rand(2,100)
b=np.concatenate((np.random.rand(1,100),4+np.random.rand(1,100)),0)
data=np.concatenate((a,b),1)
np.savetxt('data.txt',data,delimiter=',',fmt='%1.3f')
print('finished')
'''