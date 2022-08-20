"""srun -n 60 --cpu-bind=cores python pso_script.py 50 20
"""
from time import time
import argparse
from mpi4py import MPI
from jax import random as jran
import numpy as np
from mpipso import pso_update
from mpipso.tests.test_pso_update import get_random_initial_conditions

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("n_steps", type=int)
    parser.add_argument("n_dim", type=int)
    parser.add_argument("-target_seed", type=int, default=0)
    parser.add_argument("-init_seed", type=int, default=1)
    parser.add_argument("-xlo", type=float, default=0)
    parser.add_argument("-xhi", type=float, default=1)
    args = parser.parse_args()
    n_steps, n_dim = args.n_steps, args.n_dim
    xlo, xhi = args.xlo, args.xhi

    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    rank_key = jran.PRNGKey(rank + args.init_seed)
    rank_key, init_key = jran.split(rank_key, 2)
    init_cond = get_random_initial_conditions(init_key, n_dim, xlo=xlo, xhi=xhi)
    xmin, xmax, x_init, v_init, __ = init_cond

    target_key = jran.PRNGKey(args.target_seed)
    x_target = jran.uniform(target_key, minval=xmin, maxval=xmax, shape=(n_dim,))
    np.save("x_target", x_target)

    x = np.copy(x_init)
    v = np.copy(v_init)

    np.save("x_init_{}".format(rank), x_init)

    loc_dsq_init = pso_update._euclid_dsq(x_init, x_target)
    loc_dsq_best = np.copy(loc_dsq_init)
    loc_x_best = np.copy(x)
    comm.Barrier()

    swarm_x_best, swarm_dsq_best = pso_update.get_global_best(comm, x, x_target)

    loc_dsq_best_history = []
    swarm_dsq_best_history = []
    loc_x_best_history = []
    swarm_x_best_history = []
    start = time()
    for istep in range(n_steps):
        rank_key, update_key = jran.split(rank_key, 2)
        x, v = pso_update.update_particle(
            update_key, x, v, xmin, xmax, loc_dsq_best, swarm_dsq_best
        )
        istep_dsq = float(pso_update._euclid_dsq(x, x_target))
        istep_x_best, istep_dsq_best = pso_update.get_global_best(comm, x, x_target)

        if istep_dsq_best < swarm_dsq_best:
            swarm_dsq_best = istep_dsq_best
            swarm_x_best = istep_x_best

        if istep_dsq < loc_dsq_best:
            loc_dsq_best = istep_dsq
            loc_x_best = x

        loc_x_best_history.append(loc_x_best)
        loc_dsq_best_history.append(loc_dsq_best)
        swarm_x_best_history.append(swarm_x_best)
        swarm_dsq_best_history.append(swarm_dsq_best)
        comm.Barrier()

    end = time()
    runtime = end - start
    np.save("loc_x_best_history_{0}".format(rank), loc_x_best_history)
    np.save("swarm_x_best_history_{0}".format(rank), swarm_x_best_history)
    np.save("loc_dsq_best_history_{0}".format(rank), loc_dsq_best_history)
    np.save("swarm_dsq_best_history_{0}".format(rank), swarm_dsq_best_history)
