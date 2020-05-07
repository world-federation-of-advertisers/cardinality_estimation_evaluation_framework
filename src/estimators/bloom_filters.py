# Lint as: python3
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

"""Implementation of Bloom Filters and helper functions."""

import copy
import math

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators import any_sketch
from wfa_cardinality_estimation_evaluation_framework.estimators.base import DenoiserBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import NoiserBase

_UNIFORM = "uniform"
_LOG = "log"
_ANY = "any"


def invert_monotonic(f, lower=0, epsilon=0.001):
  """Inverts monotonic function f."""
  f0 = f(lower)
  def inversion(y):
    """Inverted f."""
    assert f0 <= y, (f"Positive domain inversion error."
                     f"f({lower}) = {f0}, but {y} was requested.")
    left = lower
    probe = 1
    while f(probe) < y:
      left = probe
      probe *= 2
    right = probe
    mid = (right + left) / 2
    while right - left > epsilon:
      f_mid = f(mid)
      if f_mid > y:
        right = mid
      else:
        left = mid
      mid = (right + left) / 2
    return mid
  return inversion


class BloomFilter(any_sketch.AnySketch):
  """A rough BloomFilter based on AnySketch.

  This is not an efficient representation of a bloom filter, but is good enough
  for evaluation of different sketching methods.
  """

  @classmethod
  def get_sketch_factory(cls, length, num_hashes=1):

    def f(random_seed):
      return cls(length, num_hashes, random_seed)

    return f

  def __init__(self, length, num_hashes=1, random_seed=None):
    """Creates a BloomFilter.

    Args:
       length: The length of bit vector for the bloom filter
       num_hashes: The number of hash functions to use.
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    any_sketch.AnySketch.__init__(
        self,
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.UniformDistribution(length), "dimension_1")
        ], num_hashes, [any_sketch.BitwiseOrFunction()]),
        random_seed)


class AnyDistributionBloomFilter(any_sketch.AnySketch):
  """Implement the Any Distribution Bloom Filter.

  This class allows users to use the FirstMomentEstimator with method='all' to
  estimate the cardinality. Another use case is to get a further abstraction on
  top of AnySketch to represent all BloomFilter-type sketches, so coders can
  add new cardinality estimators for the new BloomFilter-type sketches.
  """

  @classmethod
  def get_sketch_factory(cls, config):

    def f(random_seed):
      return cls(config, random_seed)

    return f

  def __init__(self, config, random_seed):
    """Create an Any Distribution Bloom Filter.

    Args:
      config: an any_sketch.SketchConfig, which include one index_specs and
        num_hashes should be 1.
      random_seed: a random seed for generating the random seeds for the hash
        functions.
    """
    assert len(config.index_specs) == 1, "Only support one distribution."
    assert config.num_hashes == 1, "Only support one hash function."
    assert isinstance(config.value_functions[0], any_sketch.BitwiseOrFunction)
    any_sketch.AnySketch.__init__(self, config, random_seed)


class UniformBloomFilter(AnyDistributionBloomFilter):
  """Implement a Uniform Bloom Filter."""

  @classmethod
  def get_sketch_factory(cls, length):

    def f(random_seed):
      return cls(length, random_seed)

    return f

  def __init__(self, length, random_seed=None):
    """Creates a BloomFilter.

    Args:
       length: The length of bit vector for the bloom filter
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    super().__init__(
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.UniformDistribution(length), "uniform")
        ], num_hashes=1, value_functions=[any_sketch.BitwiseOrFunction()]),
        random_seed)


class LogarithmicBloomFilter(AnyDistributionBloomFilter):
  """Implement an Logarithmic Bloom Filter."""

  @classmethod
  def get_sketch_factory(cls, length):

    def f(random_seed):
      return cls(length, random_seed)

    return f

  def __init__(self, length, random_seed=None):
    """Creates an LogarithmicBloomFilter.

    Args:
       length: The length of bit vector for the bloom filter.
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """

    AnyDistributionBloomFilter.__init__(
        self,
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.LogBucketDistribution(length), "log")
        ], num_hashes=1, value_functions=[any_sketch.BitwiseOrFunction()]),
        random_seed)


class UnionEstimator(EstimatorBase):
  """A class that unions BloomFilters and estimates the combined cardinality."""

  def __init__(self):
    EstimatorBase.__init__(self)

  @classmethod
  def _check_compatibility(cls, sketch_list):
    """Determines if all sketches are compatible."""
    first_sketch = sketch_list[0]
    for cur_sketch in sketch_list[1:]:
      first_sketch.assert_compatible(cur_sketch)

  @classmethod
  def union_sketches(cls, sketch_list):
    """Exposed for testing."""
    UnionEstimator._check_compatibility(sketch_list)
    union = copy.deepcopy(sketch_list[0])
    for cur_sketch in sketch_list[1:]:
      union.sketch = union.sketch + cur_sketch.sketch
    return union

  @classmethod
  def estimate_cardinality(cls, sketch):
    """Estimate the number of elements contained in the BloomFilter."""
    x = np.sum(sketch.sketch != 0)
    k = float(sketch.num_hashes())
    m = float(sketch.max_size())
    return int(math.fabs(m / k * math.log(1 - x / m)))

  def __call__(self, sketch_list):
    """Does a bit-wise of all sketches and returns a combined cardinality estimate."""
    if not sketch_list:
      return 0
    assert isinstance(sketch_list[0], BloomFilter), "expected a BloomFilter"
    union = UnionEstimator.union_sketches(sketch_list)
    return UnionEstimator.estimate_cardinality(union)


class FirstMomentEstimator(EstimatorBase):
  """First moment cardinality estimator for AnyDistributionBloomFilter."""

  def __init__(self, method, denoiser=None, weights=None):
    EstimatorBase.__init__(self)
    if denoiser is None:
      self._denoiser = copy.deepcopy
    else:
      self._denoiser = denoiser
    self._weights = weights
    assert method in (_UNIFORM, _LOG, _ANY), (
        f"method={method} not supported")
    self._method = method

  @classmethod
  def _check_compatibility(cls, sketch_list):
    """Determines if all sketches are compatible."""
    first_sketch = sketch_list[0]
    for cur_sketch in sketch_list[1:]:
      first_sketch.assert_compatible(cur_sketch)

  def union_sketches(self, sketch_list):
    """Exposed for testing."""
    FirstMomentEstimator._check_compatibility(sketch_list)
    union = self._denoiser(sketch_list[0])
    for cur_sketch in sketch_list[1:]:
      cur_sketch = self._denoiser(cur_sketch)
      union.sketch = 1 - (1 - union.sketch) * (1 - cur_sketch.sketch)
    return union

  @classmethod
  def _estimate_cardinality_uniform(cls, sketch):
    """Estimate cardinality of a Uniform Bloom Filter."""
    x = sum(sketch.sketch)
    m = len(sketch.sketch)
    return - m * math.log(1 - x / m)

  @classmethod
  def _estimate_cardinality_log(cls, sketch):
    """Estimate cardinality of an Log Bloom Filter."""
    x = sum(sketch.sketch)
    m = len(sketch.sketch)
    return x / (1 - x / m)

  @classmethod
  def _estimate_cardinality_any(cls, sketch, weights):
    """Estimate cardinality of a Bloom Filter with any distribution."""
    register_probs = sketch.config.index_specs[0].distribution.register_probs
    if weights is None:
      weights = np.ones(sketch.max_size())
    else:
      assert len(weights) == sketch.max_size()

    def first_moment(u):
      return np.dot(
          weights,
          1 - np.power(1 - register_probs, u) - sketch.sketch
      )

    lower_bound = (
        np.log(1 - np.average(sketch.sketch, weights=weights))
        / np.log(1 - np.mean(register_probs)))

    return invert_monotonic(first_moment, lower_bound)(0)

  def __call__(self, sketch_list):
    """Merge all sketches and estimates the cardinality of their union."""
    if not sketch_list:
      return 0
    assert isinstance(sketch_list[0], AnyDistributionBloomFilter), (
        "Expected an AnyDistributionBloomFilter.")
    union = self.union_sketches(sketch_list)
    if self._method == _LOG:
      return FirstMomentEstimator._estimate_cardinality_log(union)
    if self._method == _UNIFORM:
      return FirstMomentEstimator._estimate_cardinality_uniform(union)
    return FirstMomentEstimator._estimate_cardinality_any(
        union, self._weights)


class FixedProbabilityBitFlipNoiser(NoiserBase):
  """This class flips the bit of a bloom filter with a fixed probability."""

  def __init__(self, random_state, probability=None,
               flip_one_probability=None,
               flip_zero_probability=None):
    """Create a fixed probility bit flip noiser.

    Args:
      random_state: a np.random.RandomState object.
      probability: the probability that a bit will be flipped.
      flip_one_probability: the probability that a one bit will be flipped. It
        will be ignored if probability is given.
      flip_zero_probability: the probability that a zero bit will be flipped. It
        will be ignored if probability is given.
    """
    NoiserBase.__init__(self)
    if probability is not None:
      self._probability = (probability, probability)
    elif flip_one_probability is not None and flip_zero_probability is not None:
      self._probability = (flip_zero_probability, flip_one_probability)
    else:
      raise ValueError("Should provide probability or both "
                       "flip_one_probability and flip_zero_probability.")
    self._random_state = random_state

  def __call__(self, bloom_filter):
    new_filter = copy.deepcopy(bloom_filter)
    flip_probabilies = np.where(new_filter.sketch, self._probability[1],
                                self._probability[0])
    new_filter.sketch = np.where(
        self._random_state.random_sample(
            new_filter.sketch.shape) < flip_probabilies,
        np.bitwise_xor(new_filter.sketch > 0, 1),
        new_filter.sketch)
    return new_filter


class BlipNoiser(NoiserBase):
  """This class applies "Blip" noise to a BloomFilter.

  This is a common algorithm for making Bloom filters differentially private.
  See [Alaggan et. al 2012] BLIP: Non-interactive Differentially-Private
     Similarity Computation on Bloom filters
  """

  def __init__(self, epsilon, random_state):
    """Creates a Blip Perturbator.

    Args:
       epsilon: the privacy parameter
       random_state: a numpy.random.RandomState used to draw random numbers
    """
    NoiserBase.__init__(self)
    self._epsilon = epsilon
    self.random_state = random_state

  def get_probability_of_flip(self, num_hashes):
    return 1 / (1 + math.exp(self._epsilon / num_hashes))

  def __call__(self, bloom_filter):
    """Returns a copy of a BloomFilter with possibly flipped bits.

    Args:
      bloom_filter: The BloomFilter

    Returns:
      Bit flipped BloomFilter
    """
    fixed_noiser = FixedProbabilityBitFlipNoiser(
        probability=self.get_probability_of_flip(bloom_filter.num_hashes()),
        random_state=self.random_state)
    return fixed_noiser(bloom_filter)


class SurrealDenoiser(DenoiserBase):
  """A closed form denoiser for a noisy Any Distribution Bloom Filter."""

  def __init__(self, probability=None, flip_one_probability=None,
               flip_zero_probability=None):
    if probability is not None:
      self._probability = (probability, probability)
    elif flip_one_probability is not None and flip_zero_probability is not None:
      self._probability = (flip_zero_probability, flip_one_probability)
    else:
      raise ValueError("Should provide probability or both "
                       "flip_one_probability and flip_zero_probability.")

  def  __call__(self, sketch):
    return self._denoise(sketch)

  def _denoise(self, sketch):
    """Denoise a Bloom Filter.

    Args:
      sketch: a noisy Any Distribution Bloom Filter sketch.

    Returns:
      A denoised Any Distribution Bloom Filter.
    """
    denoised_sketch = copy.deepcopy(sketch)
    expected_zeros = (
        - denoised_sketch.sketch * self._probability[1]
        + (1 - denoised_sketch.sketch) * (1 - self._probability[1]))
    denoised_sketch.sketch = 1 - expected_zeros / (
        1 - self._probability[1] - self._probability[0])
    return denoised_sketch
