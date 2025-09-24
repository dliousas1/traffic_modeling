import numpy as np
import jax
import jax.numpy as jnp
from stamp_dynamics import stamp_dynamics

def eval_Jf_analytic(p):
    """
    evaluates the Jacobian analytically since the system is linear

    Inputs:
    p: list of length (n,) Parameters objects, parameters vector.
    Outputs:
    Jf: numpy array of shape (2n, 2n), Jacobian of the dynamics function with parameters p.
    """
    Jf = stamp_dynamics(p)
    return Jf
    
def eval_Jf_auto(eval_f, x0, p):
    """
    evaluates the Jacobian of the vector field f() at state x0
    p is a structure containing all model parameters
    u is the value of the input at the current time
    uses an automatic differentiation approach via JAX library

    EXAMPLES:
    Jf        = eval_Jf(eval_f,x0,p,u);
    """

    # Convert inputs to jax arrays
    x0_jax = jnp.array(x0)

    # Define a function that only depends on x for Jacobian computation
    def f_x(x):
        return jnp.array(eval_f(x, p))

    # Compute the Jacobian using JAX's jacfwd
    Jf = jax.jacfwd(f_x)(x0_jax)

    return np.array(Jf)  # Convert back to numpy array if needed