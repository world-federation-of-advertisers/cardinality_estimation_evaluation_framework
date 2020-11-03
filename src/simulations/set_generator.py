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
"""Simulation data generators for cardinality estimation.

We have implemented the following simulation data:
* Independent sets.
* Sets generated from the Exponential bow model,
* Fully-overlapped sets.
* Subsets, i.e., sets with 2 groups, one group being subset of the other.
* Sequentially correlated sets with all the previously generated ones.
* Sequentially correlated sets with the previously generated one.
"""
from absl import logging
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common.analysis import relative_error
from wfa_cardinality_estimation_evaluation_framework.common.random import choice_fast
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import SetGeneratorBase
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import _SetSizeGenerator

ORDER_ORIGINAL = 'original'
ORDER_REVERSED = 'reversed'
ORDER_RANDOM = 'random'
USER_ACTIVITY_ASSOCIATION_INDEPENDENT = 'independent'
USER_ACTIVITY_ASSOCIATION_IDENTICAL = 'identical'
CORRELATED_SETS_ALL = 'all'
CORRELATED_SETS_ONE = 'one'
# Below are optimal parameters in Exponential Bow
# (approximated by Dirac Mixture)
# see page 14, Table 1 of the paper https://research.google/pubs/pub48387/
DIRAC_MIXTURE_OPTIMAL_ALPHA = [0.164, 0.388, 0.312, 0.136]
DIRAC_MIXTURE_OPTIMAL_X = [0.065, 0.4274, 1.275, 3.140]


class IndependentSetGenerator(SetGeneratorBase):
  """Generator for independent sets."""

  @classmethod
  def get_generator_factory_with_num_and_size(cls, universe_size,
                                              num_sets, set_size):

    def f(random_state):
      return cls(universe_size, _SetSizeGenerator(num_sets, set_size),
                 random_state)

    return f

  @classmethod
  def get_generator_factory_with_set_size_list(cls, universe_size,
                                               set_size_list):

    def f(random_state):
      return cls(universe_size, set_size_list, random_state)

    return f

  def __init__(self, universe_size, set_sizes, random_state):
    self.universe_size = universe_size
    self.union_ids = set()
    self.random_state = random_state
    self.set_sizes = set_sizes

  def __iter__(self):
    for set_size in self.set_sizes:
      set_ids = choice_fast(self.universe_size, set_size, self.random_state)
      self.union_ids = self.union_ids.union(set_ids)
      yield set_ids
    return self


class ExponentialBowSetGenerator(SetGeneratorBase):
  """Generator for Exponential Bow.

  Users have different activity levels and thus have different probabilities of
  being reached.
  """

  @classmethod
  def get_generator_factory_with_num_and_size(cls, user_activity_association,
                                              universe_size,
                                              num_sets, set_size):

    def f(random_state):
      return cls(user_activity_association, universe_size,
                 _SetSizeGenerator(num_sets, set_size), random_state)

    return f

  @classmethod
  def get_generator_factory_with_set_size_list(cls, user_activity_association,
                                               universe_size, set_size_list):

    def f(random_state):
      return cls(user_activity_association, universe_size, set_size_list,
                 random_state)

    return f

  def __init__(self, user_activity_association, universe_size,
               set_sizes, random_state):
    """Initialize an Exponential Bow set generator.

    Args:
      user_activity_association: Exponential bow assumes that different users
        have different activity levels, i.e., different probabilities of being
        reached. When user_activity_association = 'identical', each user has
        the same activity level at different publishers, i.e., the most active
        users at publisher 1 are also the most active at publisher 2. When
        user_activity_association = 'independent', one's probability of being
        reached at pub 1 is totally uncorrelated with their probability of
        being reached at pub 2. Currently, other types of
        user_activity_association are not supported.
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      set_sizes: An iterator or a list containing the size of each set.
      random_state: A np.random.RandomState object.

    Raises:
      ValueError: if the given user_activity_association is not supported.
    """
    if user_activity_association == USER_ACTIVITY_ASSOCIATION_INDEPENDENT:
      self.shuffle_user = True
      # When user_activity_association = 'independent', one's probability of
      # being reached at pub 1 is totally uncorrelated with their probability
      # of being reached at pub 2.
      # It is mathematically equivalent to: users are shuffled across
      # different publishers.
    elif user_activity_association == USER_ACTIVITY_ASSOCIATION_IDENTICAL:
      self.shuffle_user = False
    else:
      raise ValueError(
          f'user_activity_association={user_activity_association} '
          'is an invalid value.')
    self.universe_size = universe_size
    self.union_ids = set()
    self.set_size_list = [set_size for set_size in set_sizes]
    self.random_state = random_state
    if min(self.set_size_list) < 50:
      raise ValueError('Too small size is not supported for Dirac bow.')

  def __iter__(self):
    universe = np.arange(self.universe_size)
    alpha = np.array(DIRAC_MIXTURE_OPTIMAL_ALPHA) * self.universe_size
    cumsum_alpha = np.concatenate([0, np.cumsum(alpha)], axis=None)
    x = DIRAC_MIXTURE_OPTIMAL_X

    def _select_ids(lb, ub, size):
      lb = int(lb)
      ub = int(ub)
      candidate_ids = np.arange(lb, ub)
      if size >= ub - lb:
        return candidate_ids
      return choice_fast(candidate_ids, size, self.random_state)

    # The actual set size generated from Exponential bow could be smaller
    # than the input set size. The following codes extract the difference
    # between the actual set size and input set size for each set, and
    # report the worst case, i.e., the case when actual set size has largest
    # relative difference to the input set size.
    worst_case_input_size = None
    worst_case_actual_size = None
    worst_case_relative_error = 0
    threshold = 0.01
    for set_size in self.set_size_list:
      reach_rate = set_size / self.universe_size
      ids = np.hstack(
          [_select_ids(cumsum_alpha[i], cumsum_alpha[i+1],
                       int(reach_rate * x[i] * alpha[i]))
           for i in range(len(alpha))])
      if self.shuffle_user:
        self.random_state.shuffle(universe)
        ids = universe[ids]
      actual_set_size = len(ids)
      re = abs(relative_error(actual_set_size, set_size))
      if re > worst_case_relative_error:
        worst_case_relative_error = re
        worst_case_input_size = set_size
        worst_case_actual_size = actual_set_size
      self.union_ids = self.union_ids.union(ids)
      yield ids
    if worst_case_relative_error > threshold:
      logging.info(
          'Actual input size is smaller than input set size.\n'
          'Worst case: input set size = %s, actual set size: %s.',
          worst_case_input_size, worst_case_actual_size)
    return self


class FullyOverlapSetGenerator(SetGeneratorBase):
  """Generator for fully overlapping sets."""

  @classmethod
  def get_generator_factory_with_num_and_size(cls, universe_size, num_sets,
                                              set_size):

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
    self.union_ids.update(choice_fast(
        self.universe_size, self.set_size, self.random_state))
    for _ in range(self.num_sets):
      yield list(self.union_ids)
    return self


class SubSetGenerator(SetGeneratorBase):
  """Generator for subsets.

  We have sets A and B with A being a subset of B. Among the list of sets we
  generate, a number of sets equal set A, while others equal set B.
  """

  @classmethod
  def get_generator_factory_with_num_and_size(cls, order, universe_size,
                                              num_large_sets, num_small_sets,
                                              large_set_size, small_set_size):

    def f(random_state):
      return cls(order, universe_size, num_large_sets, num_small_sets,
                 large_set_size, small_set_size, random_state)

    return f

  def __init__(self, order, universe_size, num_large_sets,
               num_small_sets, large_set_size, small_set_size, random_state):
    """Initialize the subset generator.

    Args:
      order: order of sets. When order = 'original', the first several sets are
         large while the later sets are small (i.e., subset of the previous
         sets).When order = 'reversed', the first several sets are small while
         the later are large. When order = 'random', we randomly shuffle the
         order of sets.
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      num_large_sets: An integer value that specifies the number of large sets
        to be generated. These large sets are fully overlapped.
      num_small_sets: An integer value that specifies the number of small sets
        to be generated. These small sets are fully overlapped and are
        contained in each large set.
      large_set_size: An integer value that specifies the size of each
        large set.
      small_set_size: An integer value that specifies the size of each
        small set.
      random_state: A np.random.RandomState object.

    Raises:
      ValueError: if the given order is not supported.
    """
    num_sets = num_large_sets + num_small_sets
    if order == ORDER_ORIGINAL:
      # Original order: large sets first
      self.set_indices = list(range(num_sets))
    elif order == ORDER_REVERSED:
      self.set_indices = list(reversed(range(num_sets)))
    elif order == ORDER_RANDOM:
      self.set_indices = random_state.choice(num_sets, num_sets, replace=False)
    else:
      raise ValueError(f'order={order} is not supported.')
    self.universe_size = universe_size
    self.num_large_sets = num_large_sets
    self.num_small_sets = num_small_sets
    assert small_set_size <= large_set_size, 'Small size must <= large size.'
    self.large_set_size = large_set_size
    self.small_set_size = small_set_size
    self.union_ids = set()
    self.random_state = random_state

  def __iter__(self):
    large_set = choice_fast(
        self.universe_size, self.large_set_size, self.random_state)
    small_set = choice_fast(
        large_set, self.small_set_size, self.random_state)
    self.union_ids.update(set(large_set))
    set_ids_list = ([large_set] * self.num_large_sets
                    + [small_set] * self.num_small_sets)
    for i in self.set_indices:
      yield set_ids_list[i]
    return self


class _SequentiallyCorrelatedAllPreviousSetGenerator(SetGeneratorBase):
  """Generator for sequentailly correlated sets.

  Each set has some overlap with the union of previously generated campaigns.
  """

  def __init__(self, shared_prop, set_size_list, random_state):
    """Construct a sequentially correlated set generator.

    Every newly generated set has some overlap with the union of the previous
    sets. The overlap is determined by the set size multiplied by the shared
    proportion. If the union is not large enough, will use the union itself.

    Args:
      shared_prop: a number between 0 and 1, indicating the proportion of ids of
        the current set coming from the union of the previously generated sets.
      set_size_list: a list of the integer numbers representing the set size of
        the sets.
      random_state: a numpy.random.RandomState instance.
    """
    self.random_state = random_state
    self.union_ids = np.array([], dtype=int)
    self.set_size_list = set_size_list
    self.num_sets = len(set_size_list)
    self.overlap_size_list = [0]
    # Find the actual size of the overlap and the total number of ids.
    # The i-th element is the size of the IDs of the i-th set to be taken from
    # the union of the previous sets.
    total_ids_size = set_size_list[0]
    for i in range(self.num_sets - 1):
      overlap_size = min(
          int(set_size_list[i + 1] * shared_prop),
          total_ids_size
      )
      self.overlap_size_list.append(overlap_size)
      total_ids_size += set_size_list[i + 1] - overlap_size
    self.ids_pool = np.arange(total_ids_size)
    self.random_state.shuffle(self.ids_pool)

  def __iter__(self):
    for i in range(len(self.set_size_list)):
      overlap_size = self.overlap_size_list[i]
      set_ids_overlapped = choice_fast(
          self.union_ids,
          overlap_size,
          self.random_state)
      set_size = self.set_size_list[i]
      set_ids_non_overlapped = self.ids_pool[:(set_size - overlap_size)]
      self.ids_pool = self.ids_pool[len(set_ids_non_overlapped):]
      self.union_ids = np.concatenate([self.union_ids, set_ids_non_overlapped])
      set_ids = np.concatenate([set_ids_overlapped, set_ids_non_overlapped])
      yield set_ids
    return self


class _SequentiallyCorrelatedThePreviousSetGenerator(SetGeneratorBase):
  """Generator for sequentailly correlated sets.

  Each set has some overlap with THE previously generated campaign.
  """

  def __init__(self, shared_prop, set_size_list, random_state):
    """Construct a sequentially correlated set generator.

    Every newly generated set has some overlap with THE previously generated
    set. The overlap is determined by the current set size multiplied by the
    shared proportion. If the previous set is not large enough, will use the
    previous set itself.

    Args:
      shared_prop: a number between 0 and 1, indicating the proportion of ids of
        the current set coming from the previously generated set. I.e.,
        overlap size divided by the current set size.
      set_size_list: a list of the integer numbers representing the set size of
        the sets.
      random_state: a numpy.random.RandomState instance.
    """
    self.random_state = random_state
    self.union_ids = np.array([], dtype=int)
    self.set_size_list = set_size_list
    self.num_sets = len(set_size_list)
    # i-th element is the overlap size for the (i+1)-st set.
    # If the previous set is not large enough, will use the previou set.
    self.overlap_size_list = [
        min(int(set_size_list[i + 1] * shared_prop), set_size_list[i])
        for i in range(self.num_sets - 1)]
    total_ids_size = int(sum(self.set_size_list) - sum(self.overlap_size_list))
    self.ids_pool = np.arange(total_ids_size)
    self.random_state.shuffle(self.ids_pool)

  def __iter__(self):
    start = 0
    for i in range(self.num_sets):
      end = start + self.set_size_list[i]
      yield self.ids_pool[start:end]
      if i < self.num_sets - 1:
        start += self.set_size_list[i] - self.overlap_size_list[i]
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
  def get_generator_factory_with_num_and_size(cls, order, correlated_sets,
                                              shared_prop, num_sets, set_size):

    def f(random_state):
      return cls(order, correlated_sets, shared_prop,
                 _SetSizeGenerator(num_sets, set_size), random_state)

    return f

  @classmethod
  def get_generator_factory_with_set_size_list(cls, order, correlated_sets,
                                               shared_prop, set_size_list):

    def f(random_state):
      return cls(order, correlated_sets, shared_prop, set_size_list,
                 random_state)

    return f

  def __init__(self, order, correlated_sets, shared_prop, set_sizes,
               random_state):
    """Initialize a sequentially correlated sets generator.

    Args:
      order: The order of the sets to be returned. It should be one of
        'original', 'reversed' and 'random'.
      correlated_sets: One of 'all' and 'one', indicating how the current set
        is correlated with the previously generated sets when the order is
        'original'.
      shared_prop: A number between 0 and 1 that specifies the proportion of ids
        which has overlaps with all the previously generated campaigns.
      set_sizes: A generator or a list containing the size of each set.
      random_state: A np.random.RandomState object.

    Raises:
      ValueError: if the given order is not supported, or if the
      correlated_sets is not supported.
    """
    self.set_size_list = [set_size for set_size in set_sizes]
    num_sets = len(self.set_size_list)
    if order == ORDER_ORIGINAL:
      self.set_indices = list(range(num_sets))
    elif order == ORDER_REVERSED:
      self.set_indices = list(reversed(range(num_sets)))
    elif order == ORDER_RANDOM:
      self.set_indices = random_state.choice(num_sets, num_sets, replace=False)
    else:
      raise ValueError(f'order={order} is not supported.')

    if correlated_sets == CORRELATED_SETS_ALL:
      self.generator = _SequentiallyCorrelatedAllPreviousSetGenerator(
          shared_prop, self.set_size_list, random_state)
    elif correlated_sets == CORRELATED_SETS_ONE:
      self.generator = _SequentiallyCorrelatedThePreviousSetGenerator(
          shared_prop, self.set_size_list, random_state)
    else:
      raise ValueError(
          f'correlated_sets={correlated_sets} is not supported.')

  def __iter__(self):
    set_ids_list = [set_ids for set_ids in self.generator]
    for i in self.set_indices:
      yield set_ids_list[i]
    return self


class DisjointSetGenerator(SetGeneratorBase):
  """Disjoint set generator.

  This set generator can be used to
  (1) evaluate cardinality estimators under the disjoint scenario,
  (2) fast test the relative error of the estimates, as it is a deterministic
    algorithm, and hence is fast.
  """

  @classmethod
  def get_generator_factory_with_set_size_list(cls, set_sizes):
    def _f(random_state):
      return cls(set_sizes, random_state)
    return _f

  @classmethod
  def get_generator_factory_with_num_and_size(cls, num_sets, set_size):
    def _f(random_state):
      return cls(_SetSizeGenerator(num_sets, set_size), random_state)
    return _f

  def __init__(self, set_sizes, random_state=None):
    """Create a disjoint set generator.

    Args:
      set_sizes: an iterable of the set sizes.
      random_state: a numpy random state instance. It is not used, but only to
        conform to the set generator constructor arguments.
    """
    self.union_ids = set()
    _ = random_state
    self.set_sizes = set_sizes
    self.start_id = 0

  def __iter__(self):
    for set_size in self.set_sizes:
      set_ids = range(self.start_id, self.start_id + set_size)
      self.union_ids = range(self.start_id+set_size)
      self.start_id += set_size
      yield set_ids
    return self
