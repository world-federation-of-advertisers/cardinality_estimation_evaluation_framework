# Copyright 2020 The Private Cardinality Estimation Framework Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulation data generators.

We have implemented the following simulation data:
* Independent sets.
* Fully-overlapped sets.
* Sequentially correlated sets with all the previously generated ones.
* Sequentially correlated sets with the previously generated one.
"""

import numpy as np

ORDER_ORIGINAL = 'original'
ORDER_REVERSED = 'reversed'
ORDER_RANDOM = 'random'
CORRELATED_SETS_ALL = 'all'
CORRELATED_SETS_ONE = 'one'


class SetGeneratorBase(object):
  """Base object for generating test sets."""

  def __next__(self):
    raise NotImplementedError()

  @classmethod
  def get_generator_factory(cls):
    """Returns a function Handle which takes a np.random.RandomState as an arg.

    This function handle, when called, will return a fully-formed SetGenerator
    object, ready to generate sets.
    """

    def f(random_state):
      _ = random_state
      raise NotImplementedError()

    _ = f
    # In an implementation, you would return f here
    # return f
    raise NotImplementedError()


class IndependentSetGenerator(SetGeneratorBase):
  """Generator for independent sets."""

  @classmethod
  def get_generator_factory(cls, universe_size, num_sets, set_size):

    def f(random_state):
      return cls(universe_size, num_sets, set_size, random_state)

    return f

  def __init__(self, universe_size, num_sets, set_size, random_state):
    self.universe_size = universe_size
    self.num_sets = num_sets
    self.set_size = set_size
    self.union_ids = set()
    self.random_state = random_state

  def __iter__(self):
    for _ in range(self.num_sets):
      set_ids = self.random_state.choice(
          self.universe_size, self.set_size, replace=False)
      self.union_ids = self.union_ids.union(set_ids)
      yield set_ids
    return self


class FullyOverlapSetGenerator(SetGeneratorBase):
  """Generator for fully overlapping sets."""

  @classmethod
  def get_generator_factory(cls, universe_size, num_sets, set_size):

    def f(random_state):
      return cls(universe_size, num_sets, set_size, random_state)

    return f

  def __init__(self, universe_size, num_sets, set_size, random_state):
    self.universe_size = universe_size
    self.num_sets = num_sets
    self.set_size = set_size
    self.union_ids = set()
    self.random_state = random_state

  def __iter__(self):
    self.union_ids.update(self.random_state.choice(
        self.universe_size, self.set_size, replace=False))
    for _ in range(self.num_sets):
      yield list(self.union_ids)
    return self


class _SequentiallyCorrelatedAllPreviousSetGenerator(SetGeneratorBase):
  """Generator for sequentailly correlated sets.

  Each set has some overlap with the union of previously generated campaigns.
  """

  def __init__(self, universe_size, num_sets, set_size, shared_prop,
               random_state):
    self.universe_size = universe_size
    self.num_sets = num_sets
    self.set_size = set_size
    self.overlap_size = int(set_size * shared_prop)
    self.random_state = random_state
    total_ids_size = int(
        set_size * num_sets - self.overlap_size * (num_sets - 1))
    self.ids_pool = self.random_state.choice(
        universe_size, total_ids_size, replace=False)
    self.union_ids = np.array([], dtype=int)

  def __iter__(self):
    for _ in range(self.num_sets):
      overlap_size = min(self.overlap_size, len(self.union_ids))
      set_ids_overlapped = self.random_state.choice(
          self.union_ids,
          overlap_size,
          replace=False)
      set_ids_non_overlapped = self.ids_pool[:(self.set_size - overlap_size)]
      self.ids_pool = self.ids_pool[len(set_ids_non_overlapped):]
      self.union_ids = np.concatenate([self.union_ids, set_ids_non_overlapped])
      set_ids = np.concatenate([set_ids_overlapped, set_ids_non_overlapped])
      yield set_ids
    return self


class _SequentiallyCorrelatedThePreviousSetGenerator(SetGeneratorBase):
  """Generator for sequentailly correlated sets.

  Each set has some overlap with THE previously generated campaign.
  """

  def __init__(self, universe_size, num_sets, set_size, shared_prop,
               random_state):
    self.universe_size = universe_size
    self.num_sets = num_sets
    self.set_size = set_size
    self.overlap_size = int(set_size * shared_prop)
    self.random_state = random_state
    total_ids_size = int(
        set_size * num_sets - self.overlap_size * (num_sets - 1))
    self.ids_pool = self.random_state.choice(
        universe_size, total_ids_size, replace=False)

  def __iter__(self):
    for i in range(self.num_sets):
      start = i * (self.set_size - self.overlap_size)
      end = start + self.set_size
      yield self.ids_pool[start:end]
    return self


class SequentiallyCorrelatedSetGenerator(SetGeneratorBase):
  """Generator for sequentially correlated sets.

  This generator can yield sequentially correlated campaigns in the original,
  reversed, or random order.
  If users set the order to be original, each time a set is generated except the
  first one, it will have a proportion of ids coming from the union of all the
  previously generated sets. The proportion is controlled by shared_prop.
  If users set the order to be reversed, then the order of the sets will be
  the reversed order of the original.
  If users set the order to be random, then the order of the sets will be
  the random order of the original.

  There are two types of correlation.
  If users specify correlated_sets='all', all sets but the first will have
  overlap with ALL the previously generated sets under order='original'.
  If users specify correlated_sets='one', all sets but the first will have
  overlap with THE previously generated set under order='original'.
  """

  @classmethod
  def get_generator_factory(cls, order, correlated_sets, universe_size,
                            num_sets, set_size, shared_prop):

    def f(random_state):
      return cls(order, correlated_sets, universe_size, num_sets, set_size,
                 shared_prop, random_state)

    return f

  def __init__(self, order, correlated_sets, universe_size, num_sets, set_size,
               shared_prop, random_state):
    """Initialize a sequentially correlated sets generator.

    Args:
      order: The order of the sets to be returned. It should be one of
        'original', 'reversed' and 'random'.
      correlated_sets: One of 'all' and 'one', indicating how the current set
        is correlated with the previously generated sets when the order is
        'original'.
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      num_sets: An integer value that specifies the number of sets to be
        generated.
      set_size: An integer value that specifies the size of each set.
      shared_prop: A number between 0 and 1 that specifies the proportion of ids
        which has overlaps with all the previously generated campaigns.
      random_state: A np.random.RandomState object.

    Raises:
      NotImplementedError: if the given order is not implemented, or if the
      correlated_sets is not implemented.
    """
    if order == ORDER_ORIGINAL:
      self.set_indices = list(range(num_sets))
    elif order == ORDER_REVERSED:
      self.set_indices = list(reversed(range(num_sets)))
    elif order == ORDER_RANDOM:
      self.set_indices = random_state.choice(num_sets, num_sets, replace=False)
    else:
      raise NotImplementedError(f'order={order} not implemented yet.')

    if correlated_sets == CORRELATED_SETS_ALL:
      self.generator = _SequentiallyCorrelatedAllPreviousSetGenerator(
          universe_size, num_sets, set_size, shared_prop, random_state)
    elif correlated_sets == CORRELATED_SETS_ONE:
      self.generator = _SequentiallyCorrelatedThePreviousSetGenerator(
          universe_size, num_sets, set_size, shared_prop, random_state)
    else:
      raise NotImplementedError(
          f'correlated_sets={correlated_sets} not implemented yet.')

  def __iter__(self):
    set_ids_list = [set_ids for set_ids in self.generator]
    for i in self.set_indices:
      yield set_ids_list[i]
    return self
