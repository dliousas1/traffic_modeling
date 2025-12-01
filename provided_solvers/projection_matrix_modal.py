import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.sparse.linalg import eigs

def projection_matrix_modal(A, B, C, q):
    """
    Given a linear time-invariant system:
        dx/dt = A x(t) + B u(t)
         y(t) = C x(t)
    this function generates a projection matrix Vq
    whose q columns are selected eigenmodes.

    Note: This version works only for single input single output systems
    i.e., when B has a single column and C a single row.

    EXAMPLE:
    Vq = projection_matrix_modal(A, B, C, q)
    """

    # Ensure inputs are sparse or dense as required
    if not issparse(A):
        A = csc_matrix(A)
    if not issparse(B):
        B = csc_matrix(B)
    if not issparse(C):
        C = csc_matrix(C)

    # Testing and converting input/output matrices to vectors
    if B.shape[1] > 1:
        print("Detected multiple inputs. This version works only for single input.")
        print("Using only the first input.")
    b = B[:, 0].toarray().flatten()  # Use the first column of B, convert to dense and 1D

    if C.shape[0] > 1:
        print("Detected multiple outputs. This version works only for single output.")
        print("Using only the first output.")
    c = C[0, :].toarray().flatten()  # Use the first row of C, convert to dense and 1D

    if issparse(A):
        A = A.toarray()

    D, V = np.linalg.eig(A)

    # Rotate B and C in eigenvector space
    b_eig = V.T @ b
    c_eig = V.T @ c

    # Compute metric |c_i * b_i / lambda_i|
    metric = np.abs(c_eig * b_eig / D)

    # Sort metric in ascending order and get indices
    sorted_indices = np.argsort(metric)

    # Select the q most important eigenvectors
    N = V.shape[1]
    Vq = V[:, sorted_indices[-q:]]
    Vq = np.fliplr(Vq)

    return Vq
