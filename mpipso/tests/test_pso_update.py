"""
"""
import numpy as np
from jax import random as jran
from ..pso_update import mc_update_velocity


TESTING_SEED = 43


def test_mc_update_velocity():
    n_dim = 2
    xmin = np.zeros(n_dim)
    xmax = np.ones(n_dim)

    ran_key = jran.PRNGKey(TESTING_SEED)
    pos_key, ran_key = jran.split(ran_key, 2)
    x = jran.uniform(pos_key, shape=(n_dim,), minval=xmin, maxval=xmax)
    v = np.zeros(n_dim) + 0.001
    b_loc = np.zeros(n_dim) + 0.5
    b_swarm = np.zeros(n_dim) + 0.5

    vnew = mc_update_velocity(ran_key, x, v, xmin, xmax, b_loc, b_swarm)
