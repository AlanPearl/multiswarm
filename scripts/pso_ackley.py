"""srun -n 60 --cpu-bind=cores python pso_ackley.py 50 20

The Ackley loss function is a 2D loss function with a global
minimum of f(0, 0) = 0, but very many local minima. It can be
extended up to arbitrary dimensionality.
"""
from time import time
import argparse
from mpi4py import MPI
# import jax
from jax import random as jran
import jax.numpy as jnp
import numpy as np
from mpipso import pso_update


def quadratic(x_array):
    return jnp.sum(x_array**2, axis=0)


def ackley(x_array):
    a = 20 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (jnp.sum(x_array**2, axis=0))))
    b = jnp.exp(0.5 * jnp.sum(jnp.cos(2*jnp.pi*x_array), axis=0))
    return 20 + jnp.e - a - b


def himelblau(x_array):
    # 2D function with four equal global minima
    assert len(x_array) == 2
    x, y = x_array
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def modified_himelblau(x_array):
    # Now, the only global minimum is at f(3, 2) = 0
    assert len(x_array) == 2
    x, y = x_array
    return himelblau(x_array) + 0.1 * ((x - 3)**2 + (y - 2)**2)


objfunc_options = {
    "quadratic": quadratic,
    "ackley": ackley,
    "himelblau": himelblau,
    "modified_himelblau": modified_himelblau,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-func", default="ackley")
    parser.add_argument("--max-nsteps", type=int, default=100)
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=20)
    # parser.add_argument("--ranks-per-particle", type=eval, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outfile", default=None)
    args = parser.parse_args()

    objfunc_name = args.objective_func.lower()
    objfunc = objfunc_options[objfunc_name]
    max_nsteps, ndim = args.max_nsteps, args.ndim
    numpart = args.num_particles
    # ranks_per_part = args.ranks_per_particle
    particles_per_rank = 1
    ranks_per_particle = 1
    xlo, xhi = -5, 5

    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    rank_key = jran.PRNGKey(1 + rank + args.seed)
    global_key = jran.PRNGKey(args.seed)
    global_key, init_key = jran.split(global_key, 2)
    init_cond = pso_update.get_lhs_initial_conditions(
        numpart, ndim, xlo=xlo, xhi=xhi, ran_key=init_key)
    xmin, xmax, x_init, v_init = init_cond
    if not rank:
        print("x_init =", x_init)
        print("v_init =", v_init)

    # This way would keep the particle dimension
    # x = x_init = np.array_split(x_init, nranks, axis=0)[rank]
    # v = v_init = np.array_split(v_init, nranks, axis=0)[rank]

    # For now, let's index out the particle dimension (aka 1 particle per rank)
    x = x_init = x_init[rank]
    v = v_init = v_init[rank]

    loc_loss_init = objfunc(x_init)
    loc_loss_best = np.copy(loc_loss_init)
    loc_x_best = np.copy(x)
    # comm.Barrier()

    swarm_x_best, swarm_loss_best = pso_update.get_global_best(
        comm, x_init, loc_loss_init)

    loc_x_history = []
    loc_v_history = []
    loc_loss_history = []
    istep = -1
    start = time()
    for istep in range(max_nsteps):
        if not rank:
            print(f"- Best loss={swarm_loss_best} at x={swarm_x_best}"
                  f" for t={istep}", flush=True)

        rank_key, update_key = jran.split(rank_key, 2)
        rank_key = update_key
        x, v = pso_update.update_particle(
            update_key, x, v, xmin, xmax, loc_x_best, swarm_x_best
        )
        istep_loss = objfunc(x)
        istep_x_best, istep_loss_best = pso_update.get_global_best(
            comm, x, istep_loss)

        if istep_loss_best < swarm_loss_best:
            swarm_loss_best = istep_loss_best
            swarm_x_best = istep_x_best

        if istep_loss < loc_loss_best:
            loc_loss_best = istep_loss
            loc_x_best = x

        loc_x_history.append(x)
        loc_v_history.append(v)
        loc_loss_history.append(istep_loss)
        # comm.Barrier()

    end = time()
    runtime = end - start
    if not rank:
        print(f"Finished {istep + 1} steps in {runtime} seconds")

    outfile = args.outfile
    if outfile is None:
        outfile = f"pso_results_{objfunc_name}.npz"

    swarm_x_history = np.array(comm.allgather(
        loc_x_history)).swapaxes(0, 1)
    swarm_v_history = np.array(comm.allgather(
        loc_v_history)).swapaxes(0, 1)
    swarm_loss_history = np.array(comm.allgather(
        loc_loss_history)).swapaxes(0, 1)

    if not rank:
        np.savez(
            outfile,
            x_histories=swarm_x_history,
            v_histories=swarm_v_history,
            loss_histories=swarm_loss_history,
            runtime=runtime
        )
