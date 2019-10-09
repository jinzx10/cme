import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('H_dia.txt')

plt.subplot(1,2,1)
plt.plot(data[:,0], data[:,1]+data[:,4])
plt.plot(data[:,0], data[:,2]+data[:,4])

plt.subplot(1,2,2)
plt.plot(data[:,0], data[:,3])

plt.show()
