"""srun -n 60 --cpu-bind=cores python pso_ackley.py 50 20

The Ackley loss function is a 2D loss function with a global
minimum of f(0, 0) = 0, but very many local minima. It can be
extended up to arbitrary dimensionality.
"""
import argparse
from mpi4py import MPI
# import jax
import jax.numpy as jnp
import numpy as np

from mpipso import ParticleSwarm


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
    parser.add_argument("--ranks-per-particle", type=int, default=None)
    parser.add_argument("--inertial", type=float, default=1.0)
    parser.add_argument("--cognitive", type=float, default=1.0)
    parser.add_argument("--social", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    objfunc_name = args.objective_func.lower()
    objfunc = objfunc_options[objfunc_name]
    max_nsteps, ndim = args.max_nsteps, args.ndim
    nparticles = args.num_particles
    ranks_per_particle = args.ranks_per_particle
    inertial = args.inertial
    cognitive = args.cognitive
    social = args.social
    xlo, xhi = -4.5, 4.5
    seed = args.seed
    outfile = args.out
    if outfile is None:
        outfile = f"pso_results_{objfunc_name}.npz"

    swarm = ParticleSwarm(
        nparticles, ndim, xlo, xhi, seed=seed, inertial_weight=inertial,
        cognitive_weight=cognitive, social_weight=social, comm=comm,
        ranks_per_particle=ranks_per_particle)

    # CASES SUPPORTED:
    # - Original use case: 3 particles on 3 ranks
    # `mpiexec -n 3 ... --num-particles 3`
    # => ranks_per_particle=1 | particles_per_rank=1
    # - 10 particles distributed across 1, 2, or 3 ranks
    # `mpiexec -n [1,2,3] ... --num-particles 10`
    # => ranks_per_particle=1 | particles_per_rank=[10,5,3-4]
    # - 10 ranks distributed across 1, 2, or 3 particles
    # `mpiexec -n 10 ... --num-particles [1,2,3]`
    # => ranks_per_particle=[10,5,3-4] | particles_per_rank=1
    # NEW CASES SUPPORTED ARE NOW SUPPORTED:
    # - Explicit intra-particle parallelization if `ranks_per_particle` is set

    results = swarm.run_pso(
        objfunc, max_nsteps)

    if not comm.rank:
        init_best_loss = results["swarm_loss_history"][0].min()
        best_loss = results["swarm_loss_history"].min()
        print(f"Initial best loss: {init_best_loss}", flush=True)
        print(f"Final best loss: {best_loss}", flush=True)
        np.savez(
            outfile,
            x_histories=results["swarm_x_history"],
            v_histories=results["swarm_v_history"],
            loss_histories=results["swarm_loss_history"],
            runtime=results["runtime"]
        )
