import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import issparse

def tgcr(M, b, tol, maxiters): 
    """
    Generalized conjugate residual method for solving Mx = b
    INPUTS
    M - matrix
    b - right hand side
    tol - convergence tolerance, terminate on norm(b - Mx) < tol * norm(b)
    maxiters - maximum number of iterations before giving up
    OUTPUTS
    x - computed solution, returns null if no convergence
    r_norms - the scaled norm of the residual at each iteration (r_norms(1) = 1)
    """

    # Generate the initial guess for x (zero)
    x = np.zeros_like(b)

    # Set the initial residual to b - Ax^0 = b
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]
    k = 0

    p_full = []
    Mp_full = [] 
    
    while (r_norms[k]/r_norms[0] > tol) & (k <= maxiters):
        # Use the residual as the first guess for the new search direction and multiply by M
        p = r.copy()
        Mp = M.dot(p)

        # Make the new Ap vector orthogonal to the previous Mp vectors,
        # and the p vectors M^TM orthogonal to the previous p vectors.        
        if k >0:
            for j in range(k): # only to the previous Mp and p due to a symmetrical M
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
        if r_norms[k] < tol * r_norms[0]:
            break
        
        k = k+1
    if r_norms[k] > tol * r_norms[0]:
        print('GCR NONCONVERGENCE!!!\n')
        x = None

    r_norms = np.array(r_norms) / r_norms[0]
    return x, r_norms
    
