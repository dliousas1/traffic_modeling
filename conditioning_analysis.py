import numpy as np;
from evaluate_f import Parameters
from evaluate_Jf import eval_Jf
from icecream import ic
import matplotlib.pyplot as plt
import scipy.sparse as sp

def conditioning_analysis(n, alpha, beta, tau, x0, u=None):
    """
    Analyzes the conditioning of the Jacobian matrix of the dynamics function f
    for a system of n cars with given parameters and initial state.

    Inputs:
    n: int, number of cars.
    alpha: float, parameter alpha for all cars.
    beta: float, parameter beta for all cars.
    tau: float, parameter tau for all cars.
    x0: numpy array of shape (2n,), initial state vector.
    u: optional input, not used in this function.

    Outputs:
    cond_Jf: float, condition number of the analytic Jacobian.
    """
    
    # Create parameters list
    p = {
        "parameters": [Parameters(alpha, beta, tau, 0, 0, 0, 0) for _ in range(n)]
    }
    
    # Evaluate analytic Jacobian
    Jf = eval_Jf(x0, p, u)
    
    # Extract the jacobian matrix for the n-1 cars that follow the lead car
    Jf = Jf[:2*(n-1), :2*(n-1)]

    # Compute condition number
    cond_Jf = compute_cond(Jf)

    return cond_Jf

def conditioning_analysis_anomaly(n, alpha, beta, tau, x0, u=None, k_anomaly=0, alpha_anomaly=None, beta_anomaly=None, tau_anomaly=None):
    """
    Analyzes the conditioning of the Jacobian matrix of the dynamics function f
    for a system of n cars with given parameters and initial state, introducing
    an anomaly in the parameters of one car.

    Inputs:
    n: int, number of cars.
    alpha: float, parameter alpha for all cars.
    beta: float, parameter beta for all cars.
    tau: float, parameter tau for all cars.
    x0: numpy array of shape (2n,), initial state vector.
    u: optional input, not used in this function.
    k_anomaly: int, index of the car to introduce the anomaly (0 <= k_anomaly < n).
    alpha_anomaly: float, anomalous parameter alpha for car k_anomaly.
    beta_anomaly: float, anomalous parameter beta for car k_anomaly.
    tau_anomaly: float, anomalous parameter tau for car k_anomaly.

    Outputs:
    cond_Jf: float, condition number of the analytic Jacobian with anomaly.
    """
    
    # Create parameters list
    p = {
        "parameters": [Parameters(alpha, beta, tau, 0, 0, 0, 0) for _ in range(n)]
    }
    
    # Introduce anomaly
    p["parameters"][k_anomaly] = Parameters(alpha_anomaly, beta_anomaly, tau_anomaly, 0, 0, 0, 0)
    
    # Evaluate analytic Jacobian
    Jf = eval_Jf(x0, p, u)
    
    # Extract the jacobian matrix for the n-1 cars that follow the lead car
    Jf = Jf[:2*(n-1), :2*(n-1)]

    # Compute condition number
    cond_Jf = compute_cond(Jf)

    return cond_Jf

   

def compute_cond(Jf):
    # # Compute condition numbers
    # cond_Jf = np.linalg.cond(Jf)

    # Make sparse matrix
    Jf = sp.csr_matrix(Jf)

    try :
        1/0
        sv_max = sp.linalg.svds(Jf, return_singular_vectors=False, k=1, which='LM')[0]
        sv_min = sp.linalg.svds(Jf, return_singular_vectors=False, k=1, which='SM')[0]
        cond_Jf = sv_max/sv_min

    except Exception as e:
        # Use numpy cond with dense matrix as fallback
        # ic(e)
        # ic("Falling back to dense matrix computation")
        Jf_dense = Jf.toarray()
        cond_Jf = np.linalg.cond(Jf_dense)
        # ic("Condition number computed with dense matrix", cond_Jf)
    
    return cond_Jf

if __name__ == "__main__":
    # Example parameters
    n_vals = np.logspace(1, 3, num=3, dtype=int)  # From 10^1 to 10^3
    alpha = 1.0
    beta = 1.0
    tau = 1.0
    # x0 = np.array([0, 0, 10, 0, 20, 0, 30, 0, 40, 3])  # Initial state for 5 cars
    cond_Jf_list = []
    for n in n_vals:
        x0 = np.zeros(2*n)  # All cars start at rest at position 0
        cond_Jf = conditioning_analysis(n, alpha, beta, tau, x0)
        cond_Jf_list.append(cond_Jf)
    ic(n_vals, cond_Jf_list)
    # ic(1e-2 * np.array(cond_Jf_list))
    # ic(1e-15 * np.array(cond_Jf_list))
    # # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.loglog(n_vals, cond_Jf_list, marker='o')
    # plt.xlabel('Number of Cars (n)')
    # plt.ylabel('Condition Number of Jacobian')
    # plt.title('Condition Number of Jacobian vs Number of Cars')
    # plt.grid(True, which="both", ls="--")
    # plt.xlim([1e1, 1e3])
    # # Make the x ticks not in scientific notation, using only the major ticks
    # plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())

    # plt.show()

    # Now with anomaly
    cond_Jf_anomaly_list = []
    alpha_anomaly = 1.0
    beta_anomaly = 1.0
    tau_anomaly = 1e5
    for n in n_vals:
        x0 = np.zeros(2*n)  # All cars start at rest at position 0
        k_anomaly = n-2  # Introduce anomaly
        cond_Jf_anomaly = conditioning_analysis_anomaly(n, alpha, beta, tau, x0, k_anomaly=k_anomaly, alpha_anomaly=alpha_anomaly, beta_anomaly=beta_anomaly, tau_anomaly=tau_anomaly)
        cond_Jf_anomaly_list.append(cond_Jf_anomaly)
    ic(n_vals, cond_Jf_anomaly_list)
    # ic(1e-2 * np.array(cond_Jf_anomaly_list))
    # ic(1e-15 * np.array(cond_Jf_anomaly_list))
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.loglog(n_vals, cond_Jf_list, marker='o', color='b', linestyle='-', label='No Anomaly')
    plt.loglog(n_vals, cond_Jf_anomaly_list, marker='o', color='r', linestyle='-', label='With Anomaly')
    plt.xlabel('Number of Cars (n)')
    plt.ylabel('Condition Number of Jacobian with Anomaly')
    plt.title('Condition Number of Jacobian vs Number of Cars with Anomaly')
    plt.grid(True, which="both", ls="--")
    plt.xlim([1e1, 1e3])
    # Make the x ticks not in scientific notation, using only the major ticks
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())  
    plt.legend()
    # plt.show()

    # For n=100, scale alpha and beta of the anomalous car (alpha=beta)
    n = 100
    x0 = np.zeros(2*n)  # All cars start at rest at position 0
    k_anomaly = n-2  # Introduce anomaly
    tau_anomaly = 1.0
    anomaly_alpha_beta_vals = np.logspace(-4, 4, num=11)  # From 10^0 to 10^5
    cond_Jf_anomaly_alpha_beta_list = []
    for anomaly_alpha_beta in anomaly_alpha_beta_vals:
        cond_Jf_anomaly = conditioning_analysis_anomaly(n, alpha, beta, tau, x0, k_anomaly=k_anomaly, alpha_anomaly=anomaly_alpha_beta, beta_anomaly=anomaly_alpha_beta, tau_anomaly=tau_anomaly)
        cond_Jf_anomaly_alpha_beta_list.append(cond_Jf_anomaly)
    ic(anomaly_alpha_beta_vals, cond_Jf_anomaly_alpha_beta_list)
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.loglog(anomaly_alpha_beta_vals, cond_Jf_anomaly_alpha_beta_list, marker='o', color='g', linestyle='-')
    plt.xlabel('Anomalous Car Parameter Alpha=Beta')
    plt.ylabel('Condition Number of Jacobian with Anomaly')
    plt.title('Condition Number of Jacobian vs Anomalous Car Parameter Alpha=Beta (n=100)')
    plt.grid(True, which="both", ls="--")
    plt.xlim([1e-4, 1e4])
    plt.ylim([1e0, 1e10])
    # Make the x ticks not in scientific notation, using only the major ticks
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.5g'))
    plt.show() 
