import numpy as np
import matplotlib.pyplot as plt


f0 = lambda x: np.power(x,2)+1
f1 = lambda x: (x-2)*(x-4)
L = lambda x,lamb: np.power(x,2)+1+lamb*(np.power(x,2)-6*x+8)
dual = lambda lamb: (-9*np.power(lamb,2))/(1+lamb) + 9

def line(X,m,n):
    Y = [x*m+n for x in X]
    return Y

X = np.linspace(0,5,100)
fob = f0(X)
fcon = f1(X)
lagr = [L(X,i) for i in range(-2,4,1)]
lamb = np.linspace(-0.9,5,100)
du = dual(lamb)

fig, ax = plt.subplots()
import IPython
IPython.embed()
ax.plot(X,fob,label ='objective')
ax.plot(X,fcon, label='constraint')
for i,la in enumerate(lagr):
    ax.plot(X,la,linestyle='--',linewidth=0.5,label='lagrangian'+str(i))
ax.grid(); ax.legend()
plt.show()
fig1, ax1 = plt.subplots()
ax1.plot(lamb,du)
plt.show()

