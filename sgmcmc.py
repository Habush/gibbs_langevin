"""Optax implementations of SGMCMC optimizers."""

"""Code adapted from the official repo of the paper 'What Are Bayesian Neural Network Posteriors Really Like?' by Izmailov et.al 2021
Corresponding file can be found at https://github.com/google-research/google-research/blob/master/bnn_hmc/core/sgmcmc.py
"""

from typing import Any, NamedTuple

import jax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import numpy as jnp
from optax import GradientTransformation
from optax import Params
from optim import *

import tree_utils

def sgld_gradient_update(step_size_fn,
                         seed,
                         momentum_decay=0.,
                         preconditioner=None):
  """Optax implementation of the SGLD optimizer.

  If momentum_decay is set to zero, we get the SGLD method [1]. Otherwise,
  we get the underdamped SGLD (SGHMC) method [2].

  Args:
    step_size_fn: a function taking training step as input and prodng the
      step size as output.
    seed: int, random seed.
    momentum_decay: float, momentum decay parameter (default: 0).
    preconditioner: Preconditioner, an object representing the preconditioner
      or None; if None, identity preconditioner is used (default: None).  [1]
        "Bayesian Learning via Stochastic Gradient Langevin Dynamics" Max
        Welling, Yee Whye Teh; ICML 2011  [2] "Stochastic Gradient Hamiltonian
        Monte Carlo" Tianqi Chen, Emily B. Fox, Carlos Guestrin; ICML 2014
  """

  if preconditioner is None:
    preconditioner = get_identity_preconditioner()

  def init_fn(params):
    return OptaxSGLDState(
        count=jnp.zeros([], jnp.int32),
        rng_key=jax.random.PRNGKey(seed),
        momentum=jax.tree_map(jnp.zeros_like, params),
        preconditioner_state=preconditioner.init(params))

  def update_fn(gradient, state, params=None):
    del params
    lr = step_size_fn(state.count)
    lr_sqrt = jnp.sqrt(lr)
    noise_std = jnp.sqrt(2 * (1 - momentum_decay))

    preconditioner_state = preconditioner.update_preconditioner(
        gradient, state.preconditioner_state)

    noise, new_key = tree_utils.normal_like_tree(gradient, state.rng_key)
    noise = preconditioner.multiply_by_m_sqrt(noise, preconditioner_state)

    def update_momentum(m, g, n):
      return momentum_decay * m + g * lr_sqrt + n * noise_std

    momentum = jax.tree_map(update_momentum, state.momentum, gradient, noise)
    updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
    updates = jax.tree_map(lambda m: m * lr_sqrt, updates)
    return updates, OptaxSGLDState(
        count=state.count + 1,
        rng_key=new_key,
        momentum=momentum,
        preconditioner_state=preconditioner_state)

  return GradientTransformation(init_fn, update_fn)
def disc_bin_sgld_gradient_update(step_size_fn, seed,
                              preconditioner=None, mh=False, temp=1.0):
    """Optax implementation of the  DULA [1] with preconditioner.
    The main difference between this and the above is that this returns the gamma (not the update) and optionally
    performs MH
    [1] - "A Langevin-like Sampler for Discrete Distributions" by Ruqi Zhang et.al 2022
    """
    EPS = 1e-10
    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(gamma):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            rng_key=jax.random.PRNGKey(seed),
            preconditioner_state=preconditioner.init(gamma),
            momentum=jnp.zeros_like(gamma))

    def update_fn(gamma, log_prob_fn, state):
        lr = step_size_fn(state.count)

        _, new_key = jax.random.split(state.rng_key)
        def approx_fn(x):
            g = jax.grad(log_prob_fn)(x)
            return -(g * ((2 * x) - 1.))
        def proposal(key, theta, step_size):
            key1, key2, key3 = jax.random.split(key, 3)

            first_term = approx_fn(theta) / temp
            second_term = 1./(2*step_size)
            diff = first_term - second_term
            # delta = jax.random.bernoulli(key, jax.nn.sigmoid(diff))
            flip_prob = jnp.exp(diff)/(1 + jnp.exp(diff))
            rr = jax.random.uniform(key1, shape=theta.shape)
            ind = (rr < flip_prob)*1.0
            theta_delta = (1 - theta)*ind + theta*(1 - ind)

            accept_prob = 1.0  # default in the unadjusted case where mh=False
            if mh:
                probs_forward = flip_prob*ind + (1 - flip_prob)*(1. - ind)
                lp_forward = jnp.sum(jnp.log(probs_forward+EPS), axis=-1)

                reverse_delta = approx_fn(theta_delta)
                diff_rev = reverse_delta - second_term
                flip_prob_rev = jnp.exp(diff_rev) / jnp.exp(1 + diff_rev)
                probs_rev = flip_prob_rev*ind + (1. - flip_prob_rev)*(1. - ind)
                lp_reverse = jnp.sum(jnp.log(probs_rev+EPS), axis=-1)

                m_term = (log_prob_fn(theta_delta) - log_prob_fn(theta)).squeeze()
                delta = jnp.exp(m_term + lp_reverse - lp_forward)
                delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
                u = jax.random.uniform(key3, shape=delta.shape)
                a = (delta > u)*1.
                accept_prob = jnp.mean(accept_prob)
                # accept_prob = jnp.clip(accept_prob, a_max=1.0)
                theta_delta = theta_delta*a + theta*(1. - a)

            return theta_delta*1.0, accept_prob


        # g = preconditioner.multiply_by_m_inv(jnp.ones_like(gradient), preconditioner_state)
        gamma, accept_prob = proposal(new_key, gamma, lr)


        return gamma, OptaxSGLDState(
            count=state.count + 1,
            preconditioner_state=None,
            rng_key=new_key,
            momentum=jnp.zeros_like(gamma)), accept_prob

    return GradientTransformation(init_fn, update_fn)

def disc_cat_sgld_gradient_update(step_size_fn, seed, dim, num_cls,
                              preconditioner=None, mh=False, temp=1.0):
    """Optax implementation of the  DULA [1] with preconditioner.

    [1] - "A Langevin-like Sampler for Discrete Distributions" by Ruqi Zhang et.al 2022
    """
    EPS = 1e-10
    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(gamma):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            rng_key=jax.random.PRNGKey(seed),
            preconditioner_state=preconditioner.init(gamma),
            momentum=jnp.zeros_like(gamma))

    def update_fn(gamma, log_prob_fn, state):
        lr = step_size_fn(state.count)

        # preconditioner_state = preconditioner.update_preconditioner(
        #     gradient, state.preconditioner_state)
        #
        # preconditioner_state = state.preconditioner_state
        _, new_key = jax.random.split(state.rng_key)

        def proposal(key, x, step_size):
            key1, key2 = jax.random.split(key, 2)

            #x_one_hot.shape = (bs, dim, num_cls)
            x_one_hot = jax.nn.one_hot(x, num_cls)*1.
            grad = jax.grad(log_prob_fn)(x_one_hot) / temp
            # grad_cur = grad[:, jnp.arange(dim), x[:,:]].squeeze(1)
            grad_cur = jax.vmap(lambda g, c: g[jnp.arange(dim), c])(grad, x)
            grad_cur = jnp.expand_dims(grad_cur, 2)
            grad_cur = jnp.tile(grad_cur, (1, 1, num_cls))
            first_term = grad - grad_cur

            second_term = jnp.ones_like(first_term)/(step_size)
            # second_term = second_term.at[:, jnp.arange(dim), x[:,:]].set(0.)
            second_term = jax.vmap(lambda g, c: g.at[jnp.arange(dim), c].set(0.))(second_term, x)
            diff = first_term - second_term
            cat_dist = tfd.Categorical(logits=diff)
            x_delta = cat_dist.sample(seed=key1)
            accept_prob = 1.0  # default in the unadjusted case where mh=False
            if mh:
                lp_forward = jnp.sum(cat_dist.log_prob(x_delta), axis=1)
                x_delta_one_hot = jax.nn.one_hot(x_delta, num_cls)*1.
                grad_delta = jax.grad(log_prob_fn)(x_delta_one_hot) / temp
                # grad_delta_cur = grad_delta[:, jnp.arange(dim), x_delta[:,:]].squeeze(1)
                grad_delta_cur = jax.vmap(lambda g, c: g[jnp.arange(dim), c])(grad_delta, x_delta)
                grad_delta_cur = jnp.expand_dims(grad_delta_cur, 2)
                grad_delta_cur = jnp.tile(grad_delta_cur, (1, 1, num_cls))
                first_term_delta =  grad_delta - grad_delta_cur

                second_term_delta = jnp.ones_like(first_term_delta) / step_size
                # second_term_delta = second_term_delta.at[:, jnp.arange(dim), x_delta[:,:]].set(0.)
                second_term_delta = jax.vmap(lambda g, c: g.at[jnp.arange(dim), c].set(0.))(second_term_delta
                                                                                            ,x_delta)

                diff_delta = first_term_delta - second_term_delta
                cat_dist_delta = tfd.Categorical(logits=diff_delta)
                lp_reverse = jnp.sum(cat_dist_delta.log_prob(x), axis=1)

                m_term = (log_prob_fn(x_delta_one_hot) - log_prob_fn(x_one_hot)).squeeze()
                la = m_term + lp_reverse - lp_forward
                u = jax.random.uniform(key2, shape=la.shape)
                la = jnp.clip(jnp.exp(la), a_max=1.0)
                accept_prob = jnp.clip(jnp.prod(jnp.sum(la)), a_max=1.)
                a = (la > u)*1
                # print(f"la: {la.shape}, x_delta: {x_delta.shape}, a: {a.shape}")
                x_delta = x_delta*a[:,None] + x*(1 - a[:,None])

            return x_delta, accept_prob


        # g = preconditioner.multiply_by_m_inv(jnp.ones_like(gradient), preconditioner_state)
        gamma, accept_prob = proposal(new_key, gamma, lr)


        return gamma, OptaxSGLDState(
            count=state.count + 1,
            preconditioner_state=None,
            rng_key=new_key,
            momentum=jnp.zeros_like(gamma)), accept_prob

    return GradientTransformation(init_fn, update_fn)

