import numpy as np
from .forward_euler import forward_euler
from .implicit import implicit
import copy
from icecream import ic
def eval_f_Shooting(x0, p, NotUsed=None):
    """
    evaluates the value of the function that ShootingNewton needs to force to zero
    
    INPUTS:
    x0        - current guess for the state that is supposed to start periodic steady state       
    p['eval_f']  - the function f(x, p, u) of the model dx/dt=f(x, p, u)
    p['eval_Jf'] - [optional] if not given Finite Difference will be used
    p['dt']      - timestep for the ODE integrator
    p['eval_u']  - the function providing the periodic input to f(x, p, u)
    NotUsed   - eval_F_Shooting evaluates input using p.eval_u
    
    OUTPUTS:
    F         - the actual value of the function to be forced to zero: F = x(T) - x0
    X         - a matrix whose columns are states at different times along the period
    t         - a vector with the corresponding times along the period.
    """
    
    # Check if p.eval_Jf is provided
    if 'eval_Jf' in p:
        FiniteDifference = 0  # use the analytical Jacobian
    else:
        FiniteDifference = 1  # compute a Finite Difference Jacobian using p.eval_f
        p['eval_Jf'] = None  # just to be sure

    t_start = 0  # for periodic steady state it does not matter when we start: 0 is convenient
    T = p['T']
    t_stop = T

    visualize = False

    # Choose an ODE integrator
    # X, t = forward_euler(p['eval_f'], x0, p, p['eval_u'], t_start, t_stop, p['dt'], visualize)
    # X, t = implicit('BackwardEuler', p['eval_f'], x0, p, p['eval_u'], t_start, t_stop, p['dt'], visualize, FiniteDifference, p['eval_Jf'])
    X, t, _ = implicit('Trapezoidal', p['eval_f'], x0, copy.deepcopy(p), p['eval_u'], t_start, t_stop, p['dt'], visualize, FiniteDifference, eval_Jf=p['eval_Jf'], use_GCR=False)

    pos_lead = x0[-2] + x0[-1] * t

    # Convert to relative positions and velocities between consecutive cars
    positions = X[0::2, :]
    # Relative positions to last car
    positions_rel = positions - pos_lead
    X[0::2, :] = positions_rel

    # Do same for initial guess
    pos0 = x0[0::2]
    pos0_rel = pos0 - pos0[-1]
    x0[0::2] = pos0_rel
    
    F = X[:, -1] - x0
    return F, X, t
