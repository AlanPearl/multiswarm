"""
"""
import numpy as np
from jax import random as jran
from ..pso_update import mc_update_velocity, update_particle, _euclid_dsq, _get_v_init


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


def test_update_particle():
    n_dim = 2
    xmin = np.zeros(n_dim)
    xmax = np.ones(n_dim)

    ran_key = jran.PRNGKey(TESTING_SEED)
    x_init_key, v_init_key, x_best_key, ran_key = jran.split(ran_key, 4)
    x = jran.uniform(x_init_key, shape=(n_dim,), minval=xmin, maxval=xmax)
    x_init = np.copy(x)
    x_target = jran.uniform(x_best_key, shape=(n_dim,), minval=xmin, maxval=xmax)
    dsq_best = _euclid_dsq(x, x_target)
    dsq_init = np.copy(dsq_best)
    v = _get_v_init(v_init_key, xmin, xmax)
    b_loc = np.copy(x)
    b_swarm = np.copy(x)

    n_updates = 500
    for istep in range(n_updates):
        x, v = update_particle(ran_key, x, v, xmin, xmax, b_loc, b_swarm)
        dsq = _euclid_dsq(x, x_target)
        if dsq < dsq_best:
            b_loc = x
            b_swarm = x
            dsq_best = dsq
        assert np.all(x >= xmin)
        assert np.all(x <= xmax)
    msg = "x_init = {0}\nx_target = {1}\nx_final={2}\nd_init={3}\nd_best={4}"
    assert False, msg.format(x_init, x_target, x, dsq_init, dsq_best)
