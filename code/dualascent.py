import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

objective = lambda x: np.power(x+1,2)+1
constraint = lambda x,a,b: x*a+b
Lagrange = lambda f0,h1,nu: f0 + nu*h1 

def hyperplane(a,b,ran):
    x1 = np.linspace(ran[0],ran[1],100)
    x2 = (float(b)/a[1])-(float(a[0])/a[1]*x1)
    return x1,x2

ran = (-3,3)
X = np.linspace(ran[0],ran[1],100)
Y = np.linspace(ran[0],ran[1],100)
[xx,yy] = np.meshgrid(X,Y)
zz = xx*xx + yy*yy
a = np.array([1.,1.])
b = -1
constr = hyperplane(a,b,ran)

#Opt matrices
P = 2*matrix([[1.0, 0.0],[0.0, 1.0]])
q = matrix([0.0,0.0])
A = matrix([1., 1.],(1,2))
bm = matrix(-1.)
sol = solvers.qp(P,q,None,None,A, bm)

#LAgrangian
#L(x,lamb) = x1^2 + x2^2 + lamb*a1*x1 + lamb*a2*x2 -lamb*b
lamb = 2
P = 2*matrix([[1.0,0.],[0.,1.]])
q = lamb*matrix([a[0],a[1]])
offset = -lamb*b
sol1 = solvers.qp(P,q)

import IPython
IPython.embed()
lamb = lamb + 0.1*(np.dot(a,np.array([sol1['x'][0],sol1['x'][1]]))-b)
P = 2*matrix([[1.0,0.],[0.,1.]])
q = lamb*matrix([a[0],a[1]])
sol1 = solvers.qp(P,q)
import IPython
IPython.embed()

plt.contourf(xx,yy,zz,20)
plt.plot(constr[0],constr[1])
plt.scatter(sol['x'][0],sol['x'][1])
plt.ylim(ran); plt.xlim(ran)
plt.show()

