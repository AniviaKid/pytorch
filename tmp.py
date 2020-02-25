import numpy as np
import matplotlib.pyplot as plt

x=[]
for i in range(0,20):
    x.append(i+1)
y=[5.89,5.83,5.84,5.86,5.89,5.92,5.95,5.97,5.99,6.02,6.04,6.06,6.07,6.09,6.11,6.12,6.14,6.15,6.16,6.18]
plt.plot(x, y)
plt.scatter(x,y)
plt.xticks(x)
plt.xlabel('Number of iterations')
plt.ylabel('valid loss')
plt.show()