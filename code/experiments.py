import matplotlib.pyplot as plt
from matplotlib import rc
from load_data import load_votes
import seaborn as sns
from ADMM import cov_ADMM
import numpy as np

def main():
    samples,files,names = load_votes()
    S = np.cov(np.array(samples).T)
    #Experiment residuals convergence respect to lambda 
    Z_lamb = []; tol_lamb = []; res_lamb = []; num_lamb = [] 
    lambdas = [0.3,0.1,0.05,0.01,0.001]
    for lamb in lambdas:
        X,Z,tol,res = cov_ADMM(S,lamb)
        Z_lamb.append(Z)
        tol_lamb.append(tol)
        res_lamb.append(res)
        num_lamb.append(len(res))

    #experiments with different values of rho
    rhos = [0.2,0.1,0.05,0.02,0.01]
    rhos = np.linspace(0.001,0.05,20)
    Z_rho = []; tol_list = []; res_list = []; num_its = [] 
    print len(rhos)
    for i in range(len(rhos)):
        X,Z,tol,res = cov_ADMM(S,0.02,rhos[i])
        Z_rho.append(Z)
        tol_list.append(tol)
        res_list.append(res)
        num_its.append(len(res[0]))
        print i

    ######### PLOTTING ##########
    rc('text',usetex=True)
    ##### Sparsity pattern against lambda
    fig,axarr = plt.subplots(2,2)
    axarr[0,0].spy(Z_lamb[0])
    axarr[0,1].spy(Z_lamb[2])
    axarr[1,0].spy(Z_lamb[3])
    axarr[1,1].spy(Z_lamb[4])
    plt.show();plt.close()
    #### residuals against lambda
    fig1,axarr1 = plt.subplots(2)
    for i in range(len(tol_lamb)):
        axarr1[0].semilogy(res_lamb[i][0], label=r'$\lambda = $'+str(lambdas[i]))
        axarr1[1].semilogy(res_lamb[i][1], label=r'$\lambda = $'+str(lambdas[i]))
    axarr1[0].legend();axarr1[1].legend()
    axarr1[0].set_ylabel(r'log $||S^k||_2$'); axarr1[1].set_ylabel(r'log $||R^k||_2$')
    axarr1[1].set_xlabel('iterations')
    sns.set();plt.show();plt.close()
    # residuals against rho
    #for i in range(len(Z_rho)):
    #    plt.semilogy(res_list[i][1][1:],label=r'$\rho$ = '+str(rhos[i]))
    #plt.ylabel(r'log $||R^k||_2$'); plt.xlabel('iterations')
    #plt.legend();sns.set();plt.show();plt.close()
    plt.plot(rhos,num_its)
    plt.xlabel(r'$\rho$');plt.ylabel('number of iterations')
    sns.set();plt.show();plt.close()


    import IPython
    IPython.embed()

if __name__ == "__main__":
    main()
