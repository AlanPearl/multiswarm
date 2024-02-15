"""Implementation of PSO algorithm described in arXiv:1108.5600 & arXiv:1310.7034"""
import numpy as np
from jax import random as jran
from mpi4py import MPI
from smt.sampling_methods import LHS

INERTIAL_WEIGHT = 0.5 / np.log(2)
ACC_CONST = 0.5 + np.log(2)
VMAX_FRAC = 0.5


def get_global_best(comm, x, loss):
    rank, nranks = comm.Get_rank(), comm.Get_size()

    rank_x_matrix = np.zeros(shape=(nranks, x.size))
    rank_x_matrix[rank, :] = x
    x_matrix = np.empty_like(rank_x_matrix)

    rank_loss_matrix = np.zeros(shape=(nranks, x.size))
    rank_loss_matrix[rank, :] = loss
    loss_matrix = np.empty_like(rank_loss_matrix)

    comm.Allreduce(rank_x_matrix, x_matrix, op=MPI.SUM)
    comm.Allreduce(rank_loss_matrix, loss_matrix, op=MPI.SUM)
    # comm.Barrier()

    loss_ranks = np.sum(loss_matrix, axis=1)
    indx_x_best = np.argmin(loss_ranks)
    x_swarm_best = x_matrix[indx_x_best, :]
    loss_swarm_best = np.min(loss_ranks)

    return x_swarm_best, loss_swarm_best


def update_particle(
    ran_key,
    x,
    v,
    xmin,
    xmax,
    b_loc,
    b_swarm,
    w=INERTIAL_WEIGHT,
    acc_loc=ACC_CONST,
    acc_swarm=ACC_CONST,
):
    xnew = x + v
    vnew = mc_update_velocity(
        ran_key, x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc, acc_swarm
    )
    xnew, vnew = _impose_reflecting_boundary_condition(xnew, vnew, xmin, xmax)
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
        x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc, acc_swarm, u_loc, u_swarm
    )


def _update_velocity_kern(
    x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc, acc_swarm, u_loc, u_swarm
):
    term1 = w * v
    term2 = u_loc * acc_loc * (b_loc - x)
    term3 = u_swarm * acc_swarm * (b_swarm - x)
    v = term1 + term2 + term3
    vmax = _get_vmax(xmin, xmax)
    v = _get_clipped_velocity(v, vmax)
    return v


def _get_vmax(xmin, xmax, vmax_frac=VMAX_FRAC):
    return vmax_frac * (xmax - xmin)


def _get_clipped_velocity(v, vmax):
    v = np.where(v > vmax, vmax, v)
    v = np.where(v < -vmax, -vmax, v)
    return v


def _get_v_init(numpart, ran_key, xmin, xmax):
    n_dim = xmin.size
    vmax = _get_vmax(xmin, xmax)
    u_init = jran.uniform(ran_key, shape=(numpart, n_dim))
    return np.array(u_init * vmax)


def _impose_reflecting_boundary_condition(x, v, xmin, xmax):
    msk_lo = x < xmin
    msk_hi = x > xmax
    x = np.where(msk_lo, xmin, x)
    x = np.where(msk_hi, xmax, x)
    v = np.where(msk_lo | msk_hi, -v, v)
    return x, v


def get_lhs_initial_conditions(numpart, ndim, xlo=0, xhi=1,
                               maximin=True, ran_key=None):
    criterion = "maximin" if maximin else "c"
    if ran_key is None:
        ran_key = jran.PRNGKey(987654321)
    xmin = np.zeros(ndim) + xlo
    xmax = np.zeros(ndim) + xhi
    xlims = np.array([xmin, xmax]).T
    x_init_key, v_init_key, ran_key = jran.split(ran_key, 3)
    x_seed = int(jran.randint(
        x_init_key, (), 0, 1000000000, dtype=np.uint32))
    sampler = LHS(xlimits=xlims, criterion=criterion, random_state=x_seed)
    x_init = sampler(numpart)
    v_init = _get_v_init(numpart, v_init_key, xmin, xmax)
    return xmin, xmax, x_init, v_init
