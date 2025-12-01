import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import issparse

def tgcr_matrix_free(fhand, xf, pf, uf, b, tolrGCR, MaxItersGCR, epsMF):
    """
    Generalized conjugate residual method for solving [df/dx] x = b 
     using a matrix-free (i.e. matrix-implicit) technique
     INPUTS
     eval_f     : name of the function that evaluates f(xf,pf,uf)
     xf         : state vector where to evaluate the Jacobian [df/dx]
     pf         : structure containing parameters used by eval_f
     uf         : input needed by eval_f
     b          : right hand side of the linear system to be solved
     tolrGCR    : convergence tolerance, terminate on norm(b - Ax) / norm(b) < tolrGCR
     MaxItersGCR: maximum number of iterations before giving up
     epsMF      : finite difference perturbation for Matrix Free directional derivative
     OUTPUTS
     x          : computed solution, returns null if no convergence
     r_norms    : vector containing ||r_k||/||r_0|| for each iteration k
    
     EXAMPLE:
     [x, r_norms] = tgcr_MatrixFree(eval_f,x0,b,tolrGCR,MaxItersGCR,epsMF)
    """

    # Generate the initial guess for x (zero)
    x = np.zeros_like(b)

    # Set the initial residual to b - Ax^0 = b
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]

    p_full = []
    Mp_full = [] 
    k = 0
    while (r_norms[k]/r_norms[0] > tolrGCR) and (k <= MaxItersGCR):
        # Use the residual as the first guess for the new search direction and multiply by M
        p = r.copy()
        epsilon=2*epsMF*np.sqrt(1+np.linalg.norm(xf,np.inf))/np.linalg.norm(p,np.inf) #NITSOL normal. great
        fepsMF  = fhand(xf+epsilon*p, pf, uf)
        fepsMF = fepsMF[0] if isinstance(fepsMF, tuple) else fepsMF
        f0 = fhand(xf, pf, uf)
        f0 = f0[0] if isinstance(f0, tuple) else f0
        Mp = (fepsMF - f0)/epsilon
        # Make the new Ap vector orthogonal to the previous Mp vectors,
        # and the p vectors M^TM orthogonal to the previous p vectors.        
        if k>0:
            for j in range(k):
                beta = np.dot(Mp, Mp_full[j])
                p -= beta * p_full[j]
                Mp -= beta * Mp_full[j]

        # Make the orthogonal Mp vector of unit length, and scale the
        # p vector so that M * p  is of unit length
        # if issparse(Ap):
        #     norm_Ap = scipy.sparse.linalg(Ap, 2)
        # else:
        #     norm_Ap = np.linalg.norm(Ap, 2)
        norm_Mp = np.linalg.norm(Mp, 2)
        Mp = Mp/norm_Mp
        p = p/norm_Mp

        p_full.append(p)
        Mp_full.append(Mp)

        # Determine the optimal amount to change x in the p direction by projecting r onto Ap
        alpha = np.dot(r, Mp)

        # Update x and r
        x = x + alpha * p
        r = r - alpha * Mp

        # Save the norm of r
        r_norms.append(np.linalg.norm(r, 2))

        # Check convergence
        if r_norms[-1] < tolrGCR * r_norms[0]:
            break
        k += 1

    if r_norms[-1] > tolrGCR * r_norms[0]:
        print('GCR NONCONVERGENCE!!!\n')
        x = None
    # else:
    #     print(f'GCR converged in {k+1} iterations')

    r_norms = np.array(r_norms) / r_norms[0]
    return x, r_norms
    
