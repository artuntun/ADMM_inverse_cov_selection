import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from mpl_toolkits.mplot3d import Axes3D

def main():

    A = matrix([[-2.0,-1.0,-1.0,0.0],[-1.0,-3.0,0.0,-1.0]])
    c = matrix([1.0,1.0])
    b = matrix([-1.0,-1.0,0.0,0.0])
    sol=solvers.lp(c,A,b)
    import IPython
    IPython.embed()

    x1 = np.linspace(-3,3,10)
    x2 = np.linspace(-3,3,10)
    X,Y = np.meshgrid(x1,x2)
    Fx = X+Y
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Fx)
    plt.show()

if __name__ == '__main__':
    main()
