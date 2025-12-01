import numpy as np
from scipy.linalg import svd
from .implicit import implicit
from .eval_f_LinearSystem import eval_f_LinearSystem
from .eval_Jf_LinearSystem import eval_Jf_LinearSystem
from .eval_u_cos import gen_eval_u_cos
import time
from scipy.sparse import issparse, csc_matrix


def projection_matrix_pod(A, B, C, q, x_start, training_data=None):
    """
    Given a linear time-invariant system:
        dx/dt = A x(t) + B u(t)
         y(t) = C x(t)
    this function generates a projection matrix Vq
    whose q columns are Principal Components of generated training input trajectories.

    Note: This version works only for single input single output
    i.e., when B has a single column and C has a single row.

    EXAMPLE:
    Vq = projection_matrix_pod(A, B, C, q)
    """

    # Ensure inputs are sparse or dense as required
    if not issparse(A):
        A = csc_matrix(A)
    if not issparse(B):
        B = csc_matrix(B)
    if not issparse(C):
        C = csc_matrix(C)

    if training_data is None:
        # Generate training data
        t_training_start = time.time()
        training_data = generate_training_data(A, B, q, x_start)
        t_training_end = time.time()
        duration_training = t_training_end - t_training_start
        print(f"Generated training data in {duration_training:.2f} seconds.")
        # Append duration to training_data for output
        training_data = training_data + (duration_training,)

    X_training, p, t_start, t_stop, dt, duration_training = training_data
    # Extract every 0.1 seconds to reduce data size
    step = int(0.1 / dt)
    X_svd = X_training[:, ::step]
    # Singular value decomposition
    t_svd_start = time.time()
    U, S, Vt = svd(X_svd)
    t_svd_end = time.time()
    duration_svd = t_svd_end - t_svd_start
    print(f"Computed SVD in {duration_svd:.2f} seconds.")

    # Use the first q left singular vectors
    Vq = U[:, :q]
    return Vq, (X_training, U, S, duration_training, duration_svd, p, t_start, t_stop, dt)


def generate_training_data(A, B, q, x_start):
    """
    Generate training data for the Principal Orthogonal Decomposition (POD).
    """

    # Testing and converting input/output matrices to vectors
    if B.shape[1] > 1:
        print("Detected multiple inputs. This version works only for single input.")
        print("Using only the first input.")
    b = B[:, 0]

    # Define parameters
    p = {'A': A, 'B': b}
    eval_f = eval_f_LinearSystem
    eval_Jf = eval_Jf_LinearSystem
    T = 5.0
    eval_u = gen_eval_u_cos(T)  # Cosine input for training
    p['T'] = T
    # Initial conditions
    t_start = 0

    t_stop = 100.0
    dt = 0.01
    visualize = False
    finite_difference = False

    # Use implicit integration (Trapezoidal method)
    X_training, t_training, _ = implicit(
        'Trapezoidal', eval_f, x_start, p, eval_u, t_start, t_stop, dt, visualize, finite_difference, eval_Jf
    )

    return X_training, p, t_start, t_stop, dt
