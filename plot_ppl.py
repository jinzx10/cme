import numpy as np
import matplotlib.pyplot as plt

state = np.loadtxt('state_E0002.txt')
x = np.loadtxt('x_E0002.txt')

state_avg = np.mean(state, axis = 1)
x_avg = np.mean(x, axis = 1)

num_1 = np.sum(x>2.6, axis=1)
num_0 = np.sum(x<=2.6, axis=1)

plt.subplot(1,2,1)
plt.plot(1-state_avg)
plt.subplot(1,2,2)
plt.plot(num_0/960)

plt.show()
#plt.savefig('state0.png', format='png', dpi=300)
