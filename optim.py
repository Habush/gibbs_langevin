# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, NamedTuple

import jax
import optax
from jax import numpy as jnp

Momentum = Any  # An arbitrary pytree of `jnp.ndarrays`
GradMomentEstimates = optax.Params  # Same type as parameters
PreconditionerState = NamedTuple  # State of a preconditioner

class OptaxSGLDState(NamedTuple):
    """Optax state for the SGLD optimizer"""
    count: jnp.ndarray
    rng_key: jnp.ndarray
    momentum: Momentum
    preconditioner_state: PreconditionerState

class Preconditioner(NamedTuple):
    """Preconditioner transformation"""
    init: Any  
    update_preconditioner: Any
    multiply_by_m_sqrt: Any
    multiply_by_m_inv: Any
    multiply_by_m_sqrt_inv: Any


class RMSPropPreconditionerState(PreconditionerState):
    grad_moment_estimates: GradMomentEstimates


def get_rmsprop_preconditioner(running_average_factor=0.99, eps=1.e-7):

    def init_fn(params):
        return RMSPropPreconditionerState(
            grad_moment_estimates=jax.tree_map(jnp.zeros_like, params))

    def update_preconditioner_fn(gradient, preconditioner_state):
        grad_moment_estimates = jax.tree_map(
            lambda e, g: e * running_average_factor + \
                         g**2 * (1 - running_average_factor),
            preconditioner_state.grad_moment_estimates, gradient)
        return RMSPropPreconditionerState(
            grad_moment_estimates=grad_moment_estimates)

    def multiply_by_m_inv_fn(vec, preconditioner_state):
        return jax.tree_map(lambda e, v: v / (eps + jnp.sqrt(e)),
                            preconditioner_state.grad_moment_estimates, vec)

    def multiply_by_m_sqrt_fn(vec, preconditioner_state):
        return jax.tree_map(lambda e, v: v * jnp.sqrt(eps + jnp.sqrt(e)),
                            preconditioner_state.grad_moment_estimates, vec)

    def multiply_by_m_sqrt_inv_fn(vec, preconditioner_state):
        return jax.tree_map(lambda e, v: v / jnp.sqrt(eps + jnp.sqrt(e)),
                            preconditioner_state.grad_moment_estimates, vec)

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)


class IdentityPreconditionerState(PreconditionerState):
    """Identity preconditioner is stateless."""


def get_identity_preconditioner():

    def init_fn(_):
        return IdentityPreconditionerState()

    def update_preconditioner_fn(*args, **kwargs):
        return IdentityPreconditionerState()

    def multiply_by_m_inv_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_inv_fn(vec, _):
        return vec

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)

def make_cyclical_lr_fn(lr_0, total, num_cycles):
    k = total // num_cycles
    def schedule_fn(step):
        rk = (step % k)
        cos_inner = jnp.pi * rk
        cos_inner /= k
        cos_out = jnp.cos(cos_inner) + 1
        lr = 0.5*cos_out*lr_0

        return lr

    return schedule_fn
