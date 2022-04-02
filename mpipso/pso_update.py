"""Implementation of PSO algorithm described in arXiv:1108.5600 & arXiv:1310.7034"""
import numpy as np
from jax import random as jran

INERTIAL_WEIGHT = 0.5 * np.log(2)
ACC_CONST = 0.5 + np.log(2)


def mc_update_velocity(
    ran_key,
    x,
    v,
    b_loc,
    b_swarm,
    w=INERTIAL_WEIGHT,
    acc_loc=ACC_CONST,
    acc_swarm=ACC_CONST,
):
    """Update the particle velocity

    Parameters
    ----------
    ran_key : jax.random.PRNGKey
        JAX random seed used to generate random speeds

    x : ndarray of shape (n_params, )
        Current position of particle

    v : ndarray of shape (n_params, )
        Current velocity of particle

    b_loc : ndarray of shape (n_params, )
        best point in history of particle

    b_swarm : ndarray of shape (n_params, )
        best point in history of swarm

    w : float, optional
        inertial weight
        Default is INERTIAL_WEIGHT defined at top of module

    acc_loc : float, optional
        local acceleration
        Default is ACC_CONST defined at top of module

    acc_swarm : float, optional
        swarm acceleration
        Default is ACC_CONST defined at top of module

    Returns
    -------
    vnew : ndarray of shape (n_params, )
        New velocity of particle

    """
    u_loc, u_swarm = jran.uniform(ran_key, shape=(2,))
    return _update_velocity_kern(
        x, v, b_loc, b_swarm, w, acc_loc, acc_swarm, u_loc, u_swarm
    )


def _update_velocity_kern(x, v, b_loc, b_swarm, w, acc_loc, acc_swarm, u_loc, u_swarm):
    term1 = w * v
    term2 = u_loc * acc_loc * (b_loc - x)
    term3 = u_swarm * acc_swarm * (b_swarm - x)
    return term1 + term2 + term3
