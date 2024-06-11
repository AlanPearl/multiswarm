"""Implementation of PSO algorithm described in arXiv:1108.5600 & arXiv:1310.7034"""
from time import time

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jran
from mpi4py import MPI
# from smt.sampling_methods import LHS
from scipy.stats import qmc

import tqdm.auto as tqdm

from mpipso.mpi_utils import split_subcomms

# INERTIAL_WEIGHT = (0.5 / np.log(2))
# ACC_CONST = (0.5 + np.log(2))
INERTIAL_WEIGHT = 1.0
COGNITIVE_WEIGHT = 0.21
SOCIAL_WEIGHT = 0.07
VMAX_FRAC = 0.4


class ParticleSwarm:
    def __init__(self, nparticles, ndim, xlow, xhigh,
                 inertial_weight=INERTIAL_WEIGHT,
                 cognitive_weight=COGNITIVE_WEIGHT,
                 social_weight=SOCIAL_WEIGHT,
                 vmax_frac=VMAX_FRAC,
                 comm=MPI.COMM_WORLD, seed=None):
        if seed is None:
            # WARNING: seed must be given explicitly in jitted functions
            seed = 0
            print(f"No seed given. Setting by default: {seed=}",
                  flush=True)
        randkey = init_randkey(seed)
        rank, nranks = comm.Get_rank(), comm.Get_size()
        if nparticles > nranks:
            particles_on_this_rank = np.array_split(
                np.arange(nparticles), nranks)[rank].tolist()
            subcomm = None
        else:
            subcomm, _, particles_on_this_rank = split_subcomms(
                nparticles, comm=comm)
            particles_on_this_rank = [particles_on_this_rank]

        num_particles_on_this_rank = len(particles_on_this_rank)
        init_key, *particle_keys = jran.split(
            randkey, nparticles + 1)
        particle_keys = [particle_keys[i] for i in particles_on_this_rank]
        init_cond = get_lhs_initial_conditions(
            nparticles, ndim, xlo=xlow, xhi=xhigh,
            vmax_frac=vmax_frac, ran_key=init_key)
        xmin, xmax, x_init, v_init = init_cond

        self.nparticles = nparticles
        self.ndim = ndim
        self.xlow, self.xhigh = xlow, xhigh
        self.comm = comm
        self.particles_on_this_rank = particles_on_this_rank
        self.num_particles_on_this_rank = num_particles_on_this_rank
        self.particle_keys = particle_keys
        self.subcomm = subcomm
        self.xmin, self.xmax = xmin, xmax
        self.x_init, self.v_init = x_init, v_init
        self.inertial_weight = inertial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.vmax_frac = vmax_frac

    def run_pso(self, objfunc, nsteps=100):
        x = [self.x_init[pr] for pr in self.particles_on_this_rank]
        v = [self.v_init[pr] for pr in self.particles_on_this_rank]

        loc_loss_best = [objfunc(xi) for xi in x]
        loc_x_best = [np.copy(xi) for xi in x]
        # comm.Barrier()

        swarm_x_best, swarm_loss_best = self.get_global_best(x, loc_loss_best)

        loc_x_history = [[] for _ in range(self.num_particles_on_this_rank)]
        loc_v_history = [[] for _ in range(self.num_particles_on_this_rank)]
        loc_loss_history = [[] for _ in range(self.num_particles_on_this_rank)]
        istep = -1
        start = time()

        def trange(x):
            if self.comm.rank:
                return range(x)
            else:
                return tqdm.trange(x, desc="PSO Progress")
        for istep in trange(nsteps):
            # if not self.comm.rank:
            #     print(f"- Best loss={swarm_loss_best} at x={swarm_x_best}"
            #           f" for t={istep}", flush=True)

            istep_loss = [None for _ in range(self.num_particles_on_this_rank)]
            for ip in range(self.num_particles_on_this_rank):
                update_key = jran.split(self.particle_keys[ip], 1)[0]
                self.particle_keys[ip] = update_key
                x[ip], v[ip] = update_particle(
                    update_key, x[ip], v[ip], self.xmin, self.xmax,
                    loc_x_best[ip], swarm_x_best, self.inertial_weight,
                    self.cognitive_weight, self.social_weight, self.vmax_frac
                )
                istep_loss[ip] = objfunc(x[ip])
            istep_x_best, istep_loss_best = self.get_global_best(x, istep_loss)

            for ip in range(self.num_particles_on_this_rank):
                if istep_loss_best <= swarm_loss_best:
                    swarm_loss_best = istep_loss_best
                    swarm_x_best = istep_x_best

                if istep_loss <= loc_loss_best:
                    loc_loss_best = istep_loss
                    loc_x_best = x

                loc_x_history[ip].append(x[ip])
                loc_v_history[ip].append(v[ip])
                loc_loss_history[ip].append(istep_loss[ip])
            # comm.Barrier()

            # anneal = annealing_frac * self.inertial_weight
            # self.inertial_weight -= anneal
            # self.social_weight += anneal

        end = time()
        runtime = end - start
        if not self.comm.rank:
            print(f"Finished {istep + 1} steps in {runtime} seconds")

        swarm_x_history = np.concatenate(self.comm.allgather(
            loc_x_history), axis=0).swapaxes(0, 1)
        swarm_v_history = np.concatenate(self.comm.allgather(
            loc_v_history), axis=0).swapaxes(0, 1)
        swarm_loss_history = np.concatenate(self.comm.allgather(
            loc_loss_history), axis=0).swapaxes(0, 1)

        return {
            "swarm_x_history": swarm_x_history,
            "swarm_v_history": swarm_v_history,
            "swarm_loss_history": swarm_loss_history,
            "runtime": runtime
        }

    def get_global_best(self, x, loss):
        all_x = np.concatenate(self.comm.allgather(x))
        all_loss = np.concatenate(self.comm.allgather(loss))
        # comm.Barrier()

        best_particle = np.argmin(all_loss)
        best_x = all_x[best_particle, :]
        best_loss = all_loss[best_particle]

        return best_x, best_loss


# def get_global_best(comm, x, loss):
#     rank, nranks = comm.Get_rank(), comm.Get_size()

#     rank_x_matrix = np.zeros(shape=(nranks, x.size))
#     rank_x_matrix[rank, :] = x
#     x_matrix = np.empty_like(rank_x_matrix)

#     rank_loss_matrix = np.zeros(shape=(nranks, x.size))
#     rank_loss_matrix[rank, :] = loss
#     loss_matrix = np.empty_like(rank_loss_matrix)

#     comm.Allreduce(rank_x_matrix, x_matrix, op=MPI.SUM)
#     comm.Allreduce(rank_loss_matrix, loss_matrix, op=MPI.SUM)
#     # comm.Barrier()

#     loss_ranks = np.sum(loss_matrix, axis=1)
#     indx_x_best = np.argmin(loss_ranks)
#     x_swarm_best = x_matrix[indx_x_best, :]
#     loss_swarm_best = np.min(loss_ranks)

#     return x_swarm_best, loss_swarm_best


def update_particle(
    ran_key,
    x,
    v,
    xmin,
    xmax,
    b_loc,
    b_swarm,
    w=INERTIAL_WEIGHT,
    acc_loc=COGNITIVE_WEIGHT,
    acc_swarm=SOCIAL_WEIGHT,
    vmax_frac=VMAX_FRAC
):
    xnew = x + v
    xnew, v = _impose_reflecting_boundary_condition(xnew, v, xmin, xmax)
    vnew = mc_update_velocity(
        ran_key, xnew, v, xmin, xmax, b_loc, b_swarm,
        w, acc_loc, acc_swarm, vmax_frac
    )
    return xnew, vnew


def mc_update_velocity(
    ran_key,
    x,
    v,
    xmin,
    xmax,
    b_loc,
    b_swarm,
    w=INERTIAL_WEIGHT,
    acc_loc=COGNITIVE_WEIGHT,
    acc_swarm=SOCIAL_WEIGHT,
    vmax_frac=VMAX_FRAC
):
    """Update the particle velocity

    Parameters
    ----------
    ran_key : jax.random.PRNGKey
        JAX random seed used to generate random speeds

    x : ndarray of shape (n_params, )
        Current position of particle

    xmin : ndarray of shape (n_params, )
        Minimum position of particle

    xmax : ndarray of shape (n_params, )
        Maximum position of particle

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
        x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc,
        acc_swarm, vmax_frac, u_loc, u_swarm)


def _update_velocity_kern(
    x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc,
    acc_swarm, vmax_frac, u_loc, u_swarm
):
    term1 = w * v
    term2 = u_loc * acc_loc * (b_loc - x)
    term3 = u_swarm * acc_swarm * (b_swarm - x)
    v = term1 + term2 + term3
    vmax = _get_vmax(xmin, xmax, vmax_frac)
    v = _get_clipped_velocity(v, vmax)
    # print(f"From x={x}: local_best={b_loc}, swarm_best={b_swarm}\n"
    #       f"v_inertia={term1}, v_cognitive={term2}, v_social={term3}",
    #       flush=True)
    return v


def _get_vmax(xmin, xmax, vmax_frac=VMAX_FRAC):
    return vmax_frac * (xmax - xmin)


def _get_clipped_velocity(v, vmax):
    # vmag = np.sqrt(np.sum(v**2))
    # if vmag > vmax:
    #     v = v * vmax / vmag
    v = np.where(v > vmax, vmax, v)
    v = np.where(v < -vmax, -vmax, v)
    return v


def _get_v_init(numpart, ran_key, xmin, xmax, vmax_frac=VMAX_FRAC):
    n_dim = xmin.size
    vmax = _get_vmax(xmin, xmax, vmax_frac)
    u_init = jran.uniform(ran_key, shape=(numpart, n_dim))
    return np.array(u_init * vmax)


def _impose_reflecting_boundary_condition(x, v, xmin, xmax):
    msk_lo = x < xmin
    msk_hi = x > xmax
    x = np.where(msk_lo, xmin, x)
    x = np.where(msk_hi, xmax, x)
    v = np.where(msk_lo | msk_hi, -v, v)
    return x, v


# def get_lhs_initial_conditions(numpart, ndim, xlo=0, xhi=1, maximin=True,
#                                vmax_frac=VMAX_FRAC, ran_key=None):
#     criterion = "maximin" if maximin else "c"
#     if ran_key is None:
#         ran_key = jran.PRNGKey(987654321)
#     xmin = np.zeros(ndim) + xlo
#     xmax = np.zeros(ndim) + xhi
#     xlims = np.array([xmin, xmax]).T
#     x_init_key, v_init_key, ran_key = jran.split(ran_key, 3)
#     x_seed = int(jran.randint(
#         x_init_key, (), 0, 1000000000, dtype=np.uint32))
#     sampler = LHS(xlimits=xlims, criterion=criterion, random_state=x_seed)
#     x_init = sampler(numpart)
#     v_init = _get_v_init(numpart, v_init_key, xmin, xmax, vmax_frac)
#     return xmin, xmax, x_init, v_init

def get_lhs_initial_conditions(numpart, ndim, xlo=0, xhi=1, random_cd=True,
                               vmax_frac=VMAX_FRAC, ran_key=None):
    opt = "random-cd" if random_cd else None
    if ran_key is None:
        ran_key = jran.PRNGKey(987654321)
    xmin = np.zeros(ndim) + xlo
    xmax = np.zeros(ndim) + xhi
    x_init_key, v_init_key = jran.split(ran_key, 2)
    x_seed = int(jran.randint(
        x_init_key, (), 0, 1000000000, dtype=np.uint32))
    sampler = qmc.LatinHypercube(ndim, optimization=opt, seed=x_seed)
    x_init = sampler.random(numpart)
    x_init = qmc.scale(x_init, xmin, xmax)
    v_init = _get_v_init(numpart, v_init_key, xmin, xmax, vmax_frac)
    return xmin, xmax, x_init, v_init


def init_randkey(randkey) -> jax.Array:
    """Check that randkey is a PRNG key or create one from an int"""
    if isinstance(randkey, int):
        randkey = jran.key(randkey)
    else:
        msg = f"Invalid {type(randkey)=}: Must be int or PRNG Key"
        assert hasattr(randkey, "dtype"), msg
        assert jnp.issubdtype(randkey.dtype, jax.dtypes.prng_key), msg

    return randkey
