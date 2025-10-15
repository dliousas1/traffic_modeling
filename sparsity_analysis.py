import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
    
from evaluate_f import Parameters
from evaluate_Jf import eval_Jf_analytic


if __name__=="__main__":
    # Set up the problem whose sparsity pattern we want to analyze
    num_cars = 30
    
    parameters = [
        Parameters(
            np.random.uniform(0.5, 1.5), 
            np.random.uniform(0.5, 1.5), 
            np.random.uniform(0.5, 1.5)
        ) for _ in range(num_cars)
    ]

    # Compute the jacobian analytically
    Jf_alternate = eval_Jf_analytic(parameters, order="alternate")
    Jf_position_first = eval_Jf_analytic(parameters, order="position_first")
    Jf_velocity_first = eval_Jf_analytic(parameters, order="velocity_first")

    # Remove rows associated with position of first car and position/velocity of last car,
    # because these cause singularities.
    Jf_alternate = np.delete(Jf_alternate, [2*(num_cars-1), 2*(num_cars-1)+1], axis=0)
    Jf_alternate = np.delete(Jf_alternate, [2*(num_cars-1), 2*(num_cars-1)+1], axis=1)

    Jf_position_first = np.delete(Jf_position_first, [num_cars-1, 2*num_cars-1], axis=0)
    Jf_position_first = np.delete(Jf_position_first, [num_cars-1, 2*num_cars-1], axis=1)

    Jf_velocity_first = np.delete(Jf_velocity_first, [2*num_cars-1, num_cars-1], axis=0)
    Jf_velocity_first = np.delete(Jf_velocity_first, [2*num_cars-1, num_cars-1], axis=1)

    # Compute the percentage of non-zero entries
    nnz = np.count_nonzero(Jf_alternate)
    total_entries = Jf_alternate.size  # All matrices have the same size

    print(f"Sparsity ratio: {nnz}/{total_entries} non-zero entries ({100*nnz/total_entries:.2f}%)")
    print("-----")

    # Compute the bandwidth of the alternate ordering
    def compute_bandwidth(matrix):
        rows, cols = matrix.nonzero()
        if len(rows) == 0:
            return 0
        return np.max(np.abs(rows - cols))
    
    bandwidth_alternate = compute_bandwidth(Jf_alternate)
    bandwidth_position_first = compute_bandwidth(Jf_position_first)
    bandwidth_velocity_first = compute_bandwidth(Jf_velocity_first)

    print(f"Bandwidth of Jf_alternate: {bandwidth_alternate}")
    print(f"Bandwidth of Jf_position_first: {bandwidth_position_first}")
    print(f"Bandwidth of Jf_velocity_first: {bandwidth_velocity_first}")
    print("-----")

    # Estimate the computational cost of LU factorization
    # O(n * bandwidth^2)
    print(f"General estimated LU factorization cost of alternate ordering: O(n * {bandwidth_alternate}^2) = O({bandwidth_alternate**2}n)")
    print(f"General estimated LU factorization cost of position_first ordering: O(n * {bandwidth_position_first}^2) = O({bandwidth_position_first**2}n)")
    print(f"General estimated LU factorization cost of velocity_first ordering: O(n * {bandwidth_velocity_first}^2) = O({bandwidth_velocity_first**2}n)")
    print("-----")

    # Estimate an upper bound on the number of fill-ins during LU factorization
    # O(bandwidth * n)
    print(f"General estimated upper bound on fill-ins during LU factorization with alternate ordering: O({bandwidth_alternate}n)")
    print(f"General estimated upper bound on fill-ins during LU factorization with position_first ordering: O({bandwidth_position_first}n)")
    print(f"General estimated upper bound on fill-ins during LU factorization with velocity_first ordering: O({bandwidth_velocity_first}n)")
    print("-----")

    # Perform LU factorization to see actual fill-ins
    lu_alternate = splu(csc_matrix(Jf_alternate),
            permc_spec="NATURAL",   # no column reordering
            diag_pivot_thresh=0.0)  # no partial pivoting (use diagonal)    
    num_fill_ins_alternate = np.count_nonzero(lu_alternate.L.toarray()) + np.count_nonzero(lu_alternate.U.toarray()) - nnz

    lu_position_first = splu(csc_matrix(Jf_position_first),
                             permc_spec="NATURAL",   # no column reordering
                             diag_pivot_thresh=0.0)  # no partial pivoting (use diagonal)
    num_fill_ins_position_first = np.count_nonzero(lu_position_first.L.toarray()) + np.count_nonzero(lu_position_first.U.toarray()) - nnz

    lu_velocity_first = splu(csc_matrix(Jf_velocity_first),
                            permc_spec="NATURAL",   # no column reordering
                            diag_pivot_thresh=0.0)  # no partial pivoting (use diagonal)
    num_fill_ins_velocity_first = np.count_nonzero(lu_velocity_first.L.toarray()) + np.count_nonzero(lu_velocity_first.U.toarray()) - nnz

    print(f"Estimated number of fill-ins for this problem during LU factorization with alternate ordering: {bandwidth_alternate * num_cars}")
    print(f"Actual number of fill-ins for this problem during LU factorization with alternate ordering: {num_fill_ins_alternate}")

    print(f"Estimated number of fill-ins for this problem during LU factorization with position_first ordering: {bandwidth_position_first * num_cars}")
    print(f"Actual number of fill-ins for this problem during LU factorization with position_first ordering: {num_fill_ins_position_first}")

    print(f"Estimated number of fill-ins for this problem during LU factorization with velocity_first ordering: {bandwidth_velocity_first * num_cars}")
    print(f"Actual number of fill-ins for this problem during LU factorization with velocity_first ordering: {num_fill_ins_velocity_first}")
    print("-----")

    # Compare the sparsity patterns
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.spy(Jf_alternate, markersize=1)
    plt.title('Alternate Ordering')
    plt.subplot(1, 3, 2)
    plt.spy(Jf_position_first, markersize=1)
    plt.title('Position First Ordering')
    plt.subplot(1, 3, 3)
    plt.spy(Jf_velocity_first, markersize=1)
    plt.title('Velocity First Ordering')

    # Compare the sparsity pattern of the best one (alternate) to identity
    plt.figure(figsize=(7, 7))
    plt.spy(Jf_alternate, markersize=3, color='blue', label='Jf_alternate')
    plt.spy(np.eye(Jf_alternate.shape[0]), markersize=1, color='red', label='Identity')
    plt.title('Alternate Ordering vs Identity Sparsity')
    plt.legend(['Jf_alternate', 'Identity'])

    plt.show()