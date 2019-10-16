import numpy as np
import matplotlib.pyplot as plt

state = np.loadtxt('x_t.txt')

state_avg = np.mean(state, axis = 1)

plt.plot(state_avg)
plt.show()
