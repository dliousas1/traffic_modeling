import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded



def tgcr(A, b, tolrGCR, MaxItersGCR):
    """
    Generalized conjugate residual method for solving Ax = b
    INPUTS
    A          : actual matrix of the linear system to be solved
    b          : right hand side
    tolrGCR    : convergence tolerance, terminate on norm(b - Ax) / norm(b) < tolrGCR
    MaxItersGCR: maximum number of iterations before giving up
    A_hat      : preconditioner matrix in banded form
    A_hat_band : tuple (lower_bandwidth, upper_bandwidth) for A_hat
    OUTPUTS
    x          : computed solution, returns null if no convergence
    r_norms    : vector containing ||r_k||/||r_0|| for each iteration k

    EXAMPLE:
    [x, r_norms] = tgcr(A,b,tol,maxiters)
    """
    # Generate the initial guess for x at itearation k=0
    x = np.zeros_like(b)

    # Set the initial residual to b - Ax^0 = b
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]

    k = 0
    p_full = []
    Ap_full = [] # np.array([])
    while (r_norms[k] / r_norms[0] > tolrGCR) and (k < MaxItersGCR):
        k += 1
        # Use the residual as the first guess for the ne search direction and computer its image
        p = r.copy()
        Ap = np.dot(A, p)

        # Make the new Ap vector orthogonal to the previous Ap vectors,
        # and the p vectors A^TA orthogonal to the previous p vectors.
        # Notice that if you know A is symmetric
        # you can save computation by limiting the for loop to just j=k-1
        # however if you need relative accuracy better than  1e-10
        # it might be safer to keep full orthogonalization even for symmetric A
        if k > 1:
            for j in range(k - 1):
                beta = np.dot(Ap, Ap_full[j])
                p -= beta * p_full[j]
                Ap -= beta * Ap_full[j]

        # Make the orthogonal Ap vector of unit length, and scale the
        # p vector so that A * p  is of unit length
        norm_Ap = np.linalg.norm(Ap, 2)
        Ap /= norm_Ap
        p /= norm_Ap

        p_full.append(p)
        Ap_full.append(Ap)

        # Determine the optimal amount to change x in the p direction by projecting r onto Ap
        alpha = np.dot(r, Ap)

        # Update x and r
        x += alpha * p
        r -= alpha * Ap

        # Save the norm of r
        r_norms.append(np.linalg.norm(r, 2))

    # if r_norms[k] > tolrGCR * r_norms[0]:
        # print('GCR did NOT converge! Maximum Number of Iterations reached')
        # x = None
    # else:
        # print(f'GCR converged in {k} iterations')

    r_norms = np.array(r_norms) / r_norms[0]
    return x, r_norms
    
# Example usage:
# x, r_norms = tgcr(A, b, tolrGCR, MaxItersGCR)

