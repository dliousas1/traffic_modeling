import numpy as np
import matplotlib.pyplot as plt
from .visualize_state import visualize_state
from .eval_Jf_FiniteDifference import eval_Jf_FiniteDifference 
from .tgcr_matrix_free import tgcr_matrix_free 
from .tgcr import tgcr

def newtonNdGCR(fhand, x0, p, u, errf, errDeltax, relDeltax, MaxIter, visualize, FiniteDifference, jfhand, tolrGCR, epsMF=None):
    """
    # uses Newton Method to solve the VECTOR nonlinear system f(x)=0
    # uses GCR to solve the linearized system at each iteration
    # x0         is the initial guess for Newton iteration
    # p          is a structure containing all parameters needed to evaluate f( )
    # u          contains values of inputs 
    # eval_f     is a text string with name of function evaluating f for a given x 
    # eval_Jf    is a text string with name of function evaluating Jacobian of f at x (i.e. derivative in 1D)
    # FiniteDifference = 1 forces the use of Finite Difference Jacobian instead of given eval_Jf
    # errF       = absolute equation error: how close do you want f to zero?
    # errDeltax  = absolute output error:   how close do you want x?
    # relDeltax  = relative output error:   how close do you want x in perentage?
    # note: 		 declares convergence if ALL three criteria are satisfied 
    # MaxItersGCR= maximum number of iterations allowed
    # visualize  = 1 shows intermediate results
    # tolrGCR   = residual tolerance target for GCR
    # epsMF     (OPTIONAL) perturbation for directional derivative for matrix-free Newton-GCR
    #
    # EXAMPLES:
    # [x,converged,errf_k,errDeltax_k,relDeltax_k,iterations] = NewtonNd(eval_f,x0,p,u,errf,errDeltax,relDeltax,MaxIter,visualize,FiniteDifference,eval_Jf,tolrGCR)
    # [x,converged,errf_k,errDeltax_k,relDeltax_k,iterations] = NewtonNd(eval_f,x0,p,u,errf,errDeltax,relDeltax,MaxIter,visualize,FiniteDifference,eval_Jf,tolrGCR,epsMF)
    """
    N = len(x0)
    MaxItersGCR = N*1.1
    k = 0
    X = np.zeros((len(x0), int(MaxItersGCR)))
    X[:, k] = x0

    f = fhand(X[:,k],p,u)
    errf_k = np.linalg.norm(f, np.inf)
    errDeltax_k = np.inf
    relDeltax_k = np.inf

    # Initialize visualization
    if visualize:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        fig.show()

    # Newton loop
    while k <= MaxIter and (errf_k > errf or errDeltax_k > errDeltax or relDeltax_k > relDeltax):
        if epsMF:
            Deltax, *_ = tgcr_matrix_free(fhand, X[:,k], p, u, -f, tolrGCR, MaxItersGCR, epsMF) # uses matrix-free
        else:
            if FiniteDifference:
                Jf = eval_Jf_FiniteDifference(X[:,k], p, u)
            else: 
                Jf = jfhand(X[:,k], p, u)
            Deltax, _ = tgcr(Jf,-f,tolrGCR,MaxItersGCR)        # gcr WITHOUT matrix-free
        X[:,k+1] = X[:,k] + Deltax
        k = k+1
        f = fhand(X[:,k],p,u)
        errf_k = np.linalg.norm(f, np.inf)
        errDeltax_k = np.linalg.norm(Deltax, np.inf)
        relDeltax_k = np.linalg.norm(Deltax,np.inf)/max(abs(X[:,k]))
        
        # Update plot
        if visualize:
            ax_top, ax_bottom = visualize_state(range(1, k + 2), X, k, '.b', ax_top, ax_bottom)
            plt.pause(0.001)

    x = X[:, k]
    iterations = k

    if errf_k<=errf and errDeltax_k<=errDeltax and relDeltax_k<=relDeltax:
        converged = True
        # print('Newton converged in {} iterations'.format(iterations))
    else:
        converged = False
        print('Newton did NOT converge! Maximum Number of Iterations reached')
    
    return x, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X[:, 0:k+1]
