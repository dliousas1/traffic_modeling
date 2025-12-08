import numpy as np


def eval_Jf(x, param_dict, u=0.0):
    """
    Computes the full Jacobian matrix J(x) for the system.
    Combines linear, nonlinear, and smooth-braking logic in one pass.
    """
    p_list = param_dict['parameters']
    n = len(p_list)
    J = np.zeros((2*n, 2*n))
    
    # Tuning parameter (MUST match eval_f)
    epsilon = 1e-1 

    for i in range(n):
        # Indices
        idx_z_i = 2 * i
        idx_v_i = 2 * i + 1
        
        # 1. Position Dynamics: dz_i/dt = v_i
        J[idx_z_i, idx_v_i] = 1.0
        
        # 2. Velocity Dynamics: dv_i/dt = a_i
        if i == n - 1:
            continue
            
        # Get state and params for this car interaction
        idx_z_j = 2 * (i + 1)
        idx_v_j = 2 * (i + 1) + 1
        
        z_i, v_i = x[idx_z_i], x[idx_v_i]
        z_j, v_j = x[idx_z_j], x[idx_v_j]
        pm = p_list[i]
        
        # Pre-compute common terms
        delta_z = z_j - z_i
        exp_term = np.exp(-pm.L * (delta_z - pm.d0))
        interaction_val = pm.K * exp_term
        
        a_phys = (pm.alpha/pm.tau) * (delta_z - interaction_val) + \
                 pm.beta * (v_j - v_i) - \
                 pm.alpha * v_i
        
        if u is not None:
             a_phys += pm.input_ampl * u

        d_inter_dzi =  pm.K * pm.L * exp_term
        d_inter_dzj = -pm.K * pm.L * exp_term
        
        da_dzi = (pm.alpha/pm.tau) * (-1.0 - d_inter_dzi)
        da_dzj = (pm.alpha/pm.tau) * ( 1.0 - d_inter_dzj)
        da_dvi = -pm.beta - pm.alpha
        da_dvj =  pm.beta

        # Smooth braking logic, enforces that we cannot brake into a negative velocity
        if a_phys < 0:
            if v_i > 0:
                arg = v_i / epsilon
                S = np.tanh(arg)
                dS_dvi = (1.0 / epsilon) * (1.0 - S**2)
            else:
                S = 0.0
                dS_dvi = 0.0 
            
            term_dzi = da_dzi * S
            term_dzj = da_dzj * S
            term_dvj = da_dvj * S
            
            term_dvi = (da_dvi * S) + (a_phys * dS_dvi)
            
        else:
            term_dzi = da_dzi
            term_dzj = da_dzj
            term_dvj = da_dvj
            term_dvi = da_dvi

        # --- C. Fill Jacobian Row ---
        J[idx_v_i, idx_z_i] = term_dzi
        J[idx_v_i, idx_z_j] = term_dzj
        J[idx_v_i, idx_v_i] = term_dvi
        J[idx_v_i, idx_v_j] = term_dvj
        
    return J