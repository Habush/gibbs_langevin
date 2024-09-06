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

# Copyright 2021 The Google Research Authors.
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
"""Utility functions for jax pytree manipulations."""

import jax
from jax import numpy as jnp


def tree_get_types(tree):
  return [p.dtype for p in jax.tree_util.tree_flatten(tree)[0]]


def tree_add(a, b):
  return jax.tree_map(lambda e1, e2: e1+e2, a, b)


def tree_diff(a, b):
  return jax.tree_map(lambda p_a, p_b: p_a - p_b, a, b)


def tree_dot(a, b):
  return sum([jnp.sum(e1 * e2) for e1, e2 in
              zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))])


def tree_dist(a, b):
  dist_sq = sum([jnp.sum((e1 - e2)**2) for e1, e2 in
                 zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))])
  return jnp.sqrt(dist_sq)


def tree_scalarmul(a, s):
  return jax.tree_map(lambda e: e*s, a)


def get_first_elem_in_sharded_tree(tree):
  return jax.tree_map(lambda p: p[0], tree)


def tree_norm(a):
  return float(jnp.sqrt(sum([jnp.sum(p_a**2) for p_a in jax.tree_util.tree_leaves(a)])))

def get_first_elem_in_sharded_tree(tree):
  return jax.tree_map(lambda p: p[0], tree)

def tree_norm(a):
  return float(jnp.sqrt(sum([jnp.sum(p_a**2) for p_a in jax.tree_util.tree_leaves(a)])))

def tree_scalarmul(a, s):
  return jax.tree_map(lambda e: e*s, a)


def normal_like_tree(a, key):
  treedef = jax.tree_util.tree_structure(a)
  num_vars = len(jax.tree_util.tree_leaves(a))
  all_keys = jax.random.split(key, num=(num_vars + 1))
  noise = jax.tree_map(lambda p, k: jax.random.normal(k, shape=p.shape), a,
                            jax.tree_util.tree_unflatten(treedef, all_keys[1:]))
  return noise, all_keys[0]


def combine_dims(a, start_dim):
  return jax.tree_util.tree_map(lambda a: a.reshape((-1,) + a.shape[start_dim:]), a)

def tree_stack(trees):
  # Source https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
  """Takes a list of trees and stacks every corresponding leaf.
  For example, given two trees ((a, b), c) and ((a', b'), c'), returns
  ((stack(a, a'), stack(b, b')), stack(c, c')).
  Useful for turning a list of objects into something you can feed to a
  vmapped function.
  """
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l) for l in grouped_leaves]
  return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
  # Source https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
  """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
  For example, given a tree ((a, b), c), where a, b, and c all have first
  dimension k, will make k trees
  [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
  Useful for turning the output of a vmapped function into normal objects.
  """
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  n_trees = leaves[0].shape[0]
  new_leaves = [[] for _ in range(n_trees)]
  for leaf in leaves:
    for i in range(n_trees):
      new_leaves[i].append(leaf[i])
  new_trees = [treedef.unflatten(l) for l in new_leaves]
  return new_trees