import numpy as np
import matplotlib.pyplot as plt
from load_data import load_votes

def computationl_covariance(cov_inv):
    """Generate samples from covariance matrix given the inverse
    Input:
        -cov_inv: covariance inverse
    Output:
        - S: computational covariance
    """
    dim = cov_inv.shape
    Linv = np.linalg.inv(np.linalg.cholesky(cov_inv))
    samples = np.random.multivariate_normal(np.zeros(dim[0]),np.eye(dim[0]),100000)
    samples = np.dot(Linv.T,samples.T)
    S = np.cov(samples)
    return S

def cov_ADMM(S,lamb,rho=0.05):
    """ADMM for computing inverse covariance matrices under the prior of
    sparsity
    Input:
        - S: computational covariance
        - lamb: regularization parameter
        - rho: learning parameter
    Output:
        - Xk: estimated inverse"""
    print "Starting estimation..."
    rel_tol=10e-3; abs_tol=10e-3
    n,m = S.shape
    Uk = generate_postivedefinite(n,m)
    Zk = generate_postivedefinite(n,m)
    Xk = generate_postivedefinite(n,m)
    maxit = 1000; it = 0; stop = False
    Xk_list = [Xk]
    Zk_list = [Zk]
    primal_tolerances = []; dual_tolerances = []
    primal_residuals = []; dual_residuals = []
    while it<maxit and stop == False:
        #estimation step
        Xk = x_update(rho,Zk,Uk,S)
        Zk = z_update(lamb,rho,Uk,Xk)
        Uk = Uk + Xk - Zk
        #caluclating tolerance and residuals
        dual_res = np.linalg.norm(rho*(Zk_list[-1]-Zk))
        primal_res = np.linalg.norm(Xk-Zk)
        dual_tol = get_dual_tol(rho,S,Xk,Uk,rel_tol,abs_tol)
        primal_tol = get_primal_tol(rho,Xk,Zk,rel_tol,abs_tol)
        if dual_res<dual_tol and primal_res < primal_tol:
            stop = True
        #saving data
        Xk_list.append(Xk)
        Zk_list.append(Zk)
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)
        dual_tolerances.append(dual_tol)
        primal_tolerances.append(primal_tol)
        it +=1
    tolerances = (dual_tolerances,primal_tolerances)
    residuals = (dual_residuals,primal_residuals)
    print "The estimation converged after {} iterations".format(it)
    return Xk,Zk,tolerances,residuals

def get_dual_tol(rho,S,X,U,rel_tol,abs_tol):
    m,n = X.shape
    tol_dual = np.sqrt(n)*abs_tol
    tol_dual += rel_tol*np.linalg.norm(S.T-np.linalg.inv(X.T)+rho*U)
    return tol_dual

def get_primal_tol(rho,X,Z,rel_tol,abs_tol):
    m,n = X.shape
    primal_tol = np.sqrt(n)*abs_tol
    primal_tol += rel_tol*max(np.linalg.norm(X),np.linalg.norm(Z))
    return primal_tol

def generate_postivedefinite(n,m):
    """Generate random positive definite matrix"""
    A = np.random.random((n,m))
    B = np.dot(A,A.T)
    return B

def soft_thresholding(kappa,a):
    """Soft thresholding operator"""
    if a>kappa:
        result = a-kappa
    elif abs(a) <= kappa:
        result = 0
    else:
        result = a + kappa
    return result

def z_update(lamb,rho,U,X):
    assert(U.shape==X.shape)
    kappa = float(lamb)/rho
    A = U+X
    Z = np.zeros_like(X)
    dim = X.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            Z[i,j] = soft_thresholding(kappa,A[i,j])
    return Z

def x_update(rho,Z,U,S):
    A = rho*(Z-U)-S
    eig,Q = np.linalg.eig(A)
    x_tilde = np.apply_along_axis(trans,0,eig,rho) 
    X_tilde = np.diagflat(x_tilde)
    X = np.dot(Q,np.dot(X_tilde,Q.T))
    return X

trans = lambda x,rho: (x + np.sqrt(np.power(x,2)+4*rho))/(2*rho)



