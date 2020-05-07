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
"""Contains the class AnySketch and the objects required to configure it."""

import collections
import sys
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common.hash_function import HashFunction
from wfa_cardinality_estimation_evaluation_framework.common.hash_function import MAX_HASH_VALUE
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase


class ValueFunction(object):
  """A base class for value function."""

  def __call__(self):
    raise NotImplementedError()

  def __eq__(self, other):
    return isinstance(other, self.__class__)


class SumFunction(ValueFunction):

  def __call__(self, x, y):
    return x + y


class BitwiseOrFunction(ValueFunction):

  def __call__(self, x, y):
    assert x in (0, 1) and y in (0, 1), (
        f'Both x and y should be binary. Got x={x}, y={y}.')
    return x | y


class Distribution(object):
  """A base class for distributions to document interface."""

  def __len__(self):
    raise NotImplementedError()

  def __eq__(self, other):
    raise NotImplementedError()

  def get_index(self, hash_value):
    raise NotImplementedError()

  @property
  def register_probs(self):
    raise NotImplementedError()


class UniformDistribution(Distribution):
  """Distributes indexes uniformly in the range [0, num_values)."""

  def __init__(self, num_values):
    """Create a truncated uniform distribution.

    Args:
      num_values: The number of values that the uniform distribuition can take
        on. This is just the modulus for the hash function.
    """
    self.num_values = num_values

  def __len__(self):
    return self.num_values

  def __eq__(self, other):
    return (isinstance(other, UniformDistribution) and
            self.num_values == other.num_values)

  def get_index(self, hash_value):
    return hash_value % self.num_values

  @property
  def register_probs(self):
    return np.ones(self.num_values) / self.num_values


class LogBucketDistribution(Distribution):
  """Distributes indexes according to Exponential-Poisson model."""

  def __init__(self, num_values):
    """Create the distribution.

    Args:
      num_values: The number of bits.
    """
    self.num_values = num_values
    self._register_probs = LogBucketDistribution.get_register_probs(
        num_values)
    self.register_bounds = LogBucketDistribution.get_register_bounds(
        self._register_probs)

  def __len__(self):
    return self.num_values

  def __eq__(self, other):
    return (isinstance(other, LogBucketDistribution) and
            self.num_values == other.num_values)

  @property
  def register_probs(self):
    return self._register_probs

  @classmethod
  def get_register_probs(cls, num_values):
    """Compute per register probability."""
    probs = -np.log((np.arange(num_values) + 1) / (num_values + 1))
    return probs / sum(probs)

  @classmethod
  def get_register_bounds(cls, register_probs):
    """Compute the right bound of the registers."""
    return np.cumsum(register_probs)

  def get_index(self, hash_value, max_hash_value=MAX_HASH_VALUE):
    index = np.searchsorted(self.register_bounds, hash_value / max_hash_value)
    return index


# distribution is one of the Distributions above.
# name is an arbitrary string that is used for debug output.
IndexSpecification = collections.namedtuple('IndexSpecification',
                                            ['distribution', 'name'])

# index_specs should be an iterable (e.g. list) of IndexSpecification.
# value_functions should be an iterable (e.g. list) of ValueFunction.
SketchConfig = collections.namedtuple('SketchConfig',
                                      ['index_specs', 'num_hashes',
                                       'value_functions'])


class AnySketch(SketchBase):
  """A generalized sketch class.

  This sketch class generalizes the data structure required to
  capture Bloom filters, HLLs, Cascading Legions, Vector of Counts, and
  other sketch types. It uses a map of register keys (tuples) to a register
  value which is a count. It is not meant to be optimized for
  performance, but is rather intended to be a flexible data
  structure that can be used for experimenting with different
  sketch types.

  The register key, which is a tuple is defined by an Index
  Specification. Each specification includes a distribution that
  describes how hashed items are to be distributed along its
  axis.

  See bloom_filter.py in this directory for an example. The
  basic Bloom filter can be seen as an Any Sketch with a single
  dimension that is distributed uniformly.
  """

  def __init__(self, config, random_seed=None, hash_function=HashFunction):
    """A flexible multi-dimensional sketch class.

    Args:
      config: a SketchConfig, see above for more about this type.
      random_seed: a random seed for generating the random seeds for the hash
        functions.
      hash_function: the hash function to use. This argument is used for
        testing.
    """
    assert len(config.value_functions) == 1, 'Now we support one ValueFunction.'
    self.config = config
    random_state = np.random.RandomState(random_seed)
    self.sketch = np.zeros(
        tuple(len(i.distribution) for i in config.index_specs),
        dtype=np.int32)
    # We create config.num_hashes * #indexes hashes. Idealy we would
    # only need one hash per index dimension, but multiple makes the
    # implementation easier. There is probably a better way that
    # allows hash bits to be consumed as we traverse the indexes.

    # This is a list of list of hash functions where the sublists
    # correspond to each "hash function" that is requested in the
    # config
    self.hash_functions = []
    for _ in range(config.num_hashes):
      self.hash_functions.append([
          hash_function(random_state.randint(sys.maxsize))
          for _ in range(len(config.index_specs))
      ])

  def num_hashes(self):
    """Get the number of times inserted items are hashed."""
    # see comment in constructor for self.hashes
    return self.config.num_hashes

  def max_size(self):
    """Returns the total number of registers.

    Returns:
      The return value is the total length of self.sketch.
    """
    size = 1
    for idx in self.config.index_specs:
      size *= len(idx.distribution)
    return size

  def get_indexes(self, x):
    """Get the index tuples that should be set for the given element."""
    indexes = []
    for index_hashes in self.hash_functions:
      combined_index = []
      for idx_spec, hash_func in zip(self.config.index_specs, index_hashes):
        combined_index.append(idx_spec.distribution.get_index(hash_func(x)))
      indexes.append(tuple(combined_index))
    return indexes

  def add(self, x):
    indexes = self.get_indexes(x)
    # Move this to the loop when we support more than one value function.
    value_function = self.config.value_functions[0]
    for index in indexes:
      self.sketch[index] = value_function(self.sketch[index], 1)

  def __contains__(self, x):
    """Check for presence of an item in the Sketch.

    Args:
      x: what to check for.

    Returns:
      True if x may have been inserted, False if it is not present.
      This method may return false positives.
    """
    indexes = self.get_indexes(x)
    return self.sketch[indexes] > 0

  def assert_compatible(self, other):
    """Assert that other is compatible."""
    assert self.config == other.config, ('configs are not the same self: %s '
                                         'other %s') % (self.config,
                                                        other.config)

    assert self.hash_functions == other.hash_functions, (
        'hash functions are not the same')
    return True
