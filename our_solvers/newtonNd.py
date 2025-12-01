import numpy as np
from typing import Literal
from scipy.linalg import solve_banded

from provided_solvers.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference
from our_solvers.tgcr import tgcr


def to_banded(A, l, u):
    """
    Convert a dense square matrix A into the banded form expected by scipy.linalg.solve_banded.
    """
    A = np.asarray(A)
    n = A.shape[0]
    ab = np.zeros((l + u + 1, n))

    # Fill diagonals: row i in ab corresponds to diagonal (i - u)
    for i in range(-l, u + 1):
        diag = np.diagonal(A, offset=i)
        row_index = u - i  # shift so main diagonal (i=0) maps to row u
        ab[row_index, max(0, i):n + min(0, i)] = diag

    return ab

def newtonNd(fhand, x0, p, u,errf,errDeltax,relDeltax,MaxIter, FiniteDifference, Jfhand, Jf_bandwidth, linearSolver: Literal["LU", "solve_banded", "tgcr"] = "LU"):
    """
    # uses Newton Method to solve the VECTOR nonlinear system f(x)=0.
    # uses a banded solver to solve the linear system at each Newton iteration
    # 
    # INPUTS: 
    # x0        is the initial guess for Newton iteration
    # p         is a structure containing all parameters needed to evaluate f( )
    # u         contains values of inputs 
    # eval_f    is a text string with name of function evaluating f for a given x 
    # eval_Jf   is a text string with name of function evaluating Jacobian of f at x (i.e. derivative in 1D)
    # FiniteDifference = 1 forces the use of Finite Difference Jacobian instead of given eval_Jf
    # errF      = absolute equation error: how close do you want f to zero?
    # errDeltax = absolute output error:   how close do you want x?
    # relDeltax = relative output error:   how close do you want x in perentage?
    # note: 		declares convergence if ALL three criteria are satisfied 
    # MaxIter   = maximum number of iterations allowed
    #
    # OUTPUTS:
    # converged   1 if converged, 0 if not converged
    # errf_k      ||f(x)||
    # errDeltax_k ||X(end) -X(end-1)||
    # relDeltax_k ||X(end) -X(end-1)|| / ||X(end)||
    # iterations  number of Newton iterations k to get to convergence
    #
    # EXAMPLE:
    # x,converged,errf_k,errDeltax_k,relDeltax_k,iterations = newtonNd(eval_f,x0,p,u,errf,errDeltax,relDeltax,MaxIter,FiniteDifference,eval_Jf)
    """

    k = 0                        # Newton iteration index
    X = np.zeros((len(x0), MaxIter+1))
    X[:,k] = x0                        # X stores intermetiade solutions as columns

    f = fhand(X[:,k],p,u)
    errf_k  = np.linalg.norm(f, np.inf)

    errDeltax_k = np.float32('inf')
    relDeltax_k = np.float32('inf')

    while k < MaxIter and (errf_k > errf or errDeltax_k > errDeltax or relDeltax_k > relDeltax):

        if FiniteDifference:
            Jf,_ = eval_Jf_FiniteDifference(fhand,X[:,k],p,u)
        else: 
            Jf = Jfhand(X[:,k],p,u)

        if linearSolver == "tgcr":
            Deltax, _ = tgcr(Jf, -f, 1e-8, 1000)
        elif linearSolver == "solve_banded":
            Jf_banded = to_banded(Jf, *Jf_bandwidth)
            Deltax = solve_banded(Jf_bandwidth, Jf_banded, -f)
        else:  # linearSolver == "LU"
            Deltax = np.linalg.solve(Jf, -f)

        X[:, k+1] = X[:,k] + Deltax
        k = k+1
        f = fhand(X[:,k],p,u)
        errf_k = np.linalg.norm(f, np.inf)
        errDeltax_k = np.linalg.norm(Deltax, np.inf)
        relDeltax_k = np.linalg.norm(Deltax, np.inf)/max(abs(X[:,k]))
        
    # returning the number of iterations with ACTUAL computation
    # i.e. exclusing the given initial guess
    iterations = k 

    if errf_k <=errf and errDeltax_k<=errDeltax and relDeltax_k<=relDeltax:
        converged = True
    else:
        converged = False
        print('Newton did NOT converge! Maximum Number of Iterations reached')
    
    return X[:, k], converged, errf_k, errDeltax_k, relDeltax_k, iterations, X[:, 0:k+1]