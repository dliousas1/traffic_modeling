import numpy as np
from scipy.linalg import svd
from implicit import implicit
from eval_f_LinearSystem import eval_f_LinearSystem
from eval_Jf_LinearSystem import eval_Jf_LinearSystem
from eval_u_cos import eval_u_cos

def projection_matrix_pod(A, B, C, q):
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

    # Generate training data
    X_training = generate_training_data(A, B, q)

    # Singular value decomposition
    U, S, Vt = svd(X_training)

    # Use the first q left singular vectors
    Vq = U[:, :q]
    return Vq


def generate_training_data(A, B, q):
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
    eval_u = eval_u_cos  # Cosine input for training

    # Initial conditions
    t_start = 0
    x_start = np.ones(A.shape[1])  # Initial state vector

    t_stop = 1
    dt = 0.1
    visualize = False
    finite_difference = False

    # Use implicit integration (Trapezoidal method)
    X_training, t_training, _ = implicit(
        'Trapezoidal', eval_f, x_start, p, eval_u, t_start, t_stop, dt, visualize, finite_difference, eval_Jf
    )

    return X_training
