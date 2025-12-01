import numpy as np
from .projection_matrix_modal import projection_matrix_modal
from .projection_matrix_pod import projection_matrix_pod
from icecream import ic
def reduce_system(A, B, C, x_start, q, method, training_data=None):
    """
    Given a linear time-invariant system:
        dx/dt = A x(t) + B u(t)
         y(t) = C x(t)
    this function generates matrices describing a reduced model of specified order q:
        dxr/dt = Ar xr(t) + Br u(t)
         yr(t) = Cr xr(t)
    as well as the reduced initial state xr_start and the projection matrix Vq
    using the algorithm specified by method.

    Note: At the moment, the available methods are 'POD' and 'MODAL'.

    EXAMPLE:
    Vq, Ar, Br, Cr, xr_start = reduce_system(A, B, C, x_start, q, method)
    """

    # Determine the projection matrix Vq based on the specified method
    if method == 'MODAL':
        Vq = projection_matrix_modal(A, B, C, q)  # Use the previously defined function
    elif method == 'POD':
        Vq, extra_outputs = projection_matrix_pod(A, B, C, q, x_start, training_data=training_data)  # Use the previously defined function
    else:
        print("Method not available: returning system not reduced")
        Vq = np.eye(B.shape[0])  # Return an identity matrix (no reduction)

    # Reduce the system equations to q x q
    Ar = Vq.T @ A @ Vq
    Br = Vq.T @ B
    Cr = C @ Vq

    # Compute the reduced initial state
    xr_start = Vq.T @ x_start

    return Vq, Ar, Br, Cr, xr_start, extra_outputs
