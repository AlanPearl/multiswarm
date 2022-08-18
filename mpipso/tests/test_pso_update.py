"""
"""
import pytest
import numpy as np
from jax import random as jran
from .. import pso_update


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

    vnew = pso_update.mc_update_velocity(ran_key, x, v, xmin, xmax, b_loc, b_swarm)
    vmax = pso_update._get_vmax(xmin, xmax)
    assert np.all(vmax > 0)
    assert np.all(np.abs(vnew) <= vmax)


def test_get_v_init():
    n_dim = 2

    LO, HI = -100, 200
    ran_key = jran.PRNGKey(TESTING_SEED)
    n_tests = 50
    for itest in range(n_tests):
        v_init_key, xlo_key, xhi_key, ran_key = jran.split(ran_key, 4)
        xmin = jran.uniform(xlo_key, shape=(n_dim,), minval=LO, maxval=HI)
        delta = HI - xmin
        dx = jran.uniform(xlo_key, shape=(n_dim,), minval=0, maxval=delta)
        xmax = xmin + dx
        assert np.all(LO < xmin)
        assert np.all(xmin < xmax)
        assert np.all(xmax < HI)

        v_init = pso_update._get_v_init(v_init_key, xmin, xmax)
        vmax = pso_update._get_vmax(xmin, xmax)
        assert np.all(np.abs(v_init) < vmax)


def test_update_single_particle():
    n_dim = 2
    xmin = np.zeros(n_dim)
    xmax = np.ones(n_dim)

    ran_key = jran.PRNGKey(TESTING_SEED)
    x_init_key, v_init_key, x_best_key, ran_key = jran.split(ran_key, 4)
    x_init = jran.uniform(x_init_key, shape=(n_dim,), minval=xmin, maxval=xmax)
    x = np.copy(x_init)

    x_target = jran.uniform(x_best_key, shape=(n_dim,), minval=xmin, maxval=xmax)

    dsq_best = pso_update._euclid_dsq(x, x_target)
    dsq_init = np.copy(dsq_best)

    v_init = pso_update._get_v_init(v_init_key, xmin, xmax)
    b_loc = np.copy(x)
    b_swarm = np.copy(x)
    v = np.copy(v_init)

    n_updates = 500
    for istep in range(n_updates):
        x, v = pso_update.update_particle(ran_key, x, v, xmin, xmax, b_loc, b_swarm)
        dsq = pso_update._euclid_dsq(x, x_target)
        if dsq < dsq_best:
            b_loc = x
            b_swarm = x
            dsq_best = dsq
        assert np.all(x >= xmin), "x = {0} is below xmin"
        assert np.all(x <= xmax), "x = {0} is above xmax"

    if dsq_best == dsq_init:
        msg_prefix = "Best value did not improve initial value\n"
    elif dsq_best > dsq_init:
        msg_prefix = "Best value got worse than initial value\n"
    msg = msg_prefix + "x_init = {0}\nx_target = {1}\n"
    msg = msg + "x_final = {2}\nd_init = {3}\nd_best = {4}"
    args = x_init, x_target, x, dsq_init, dsq_best
    assert dsq_best < dsq_init, msg.format(*args)
