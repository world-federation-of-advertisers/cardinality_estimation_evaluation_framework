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
from scipy import special

from wfa_cardinality_estimation_evaluation_framework.estimators import any_sketch
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import GeometricEstimateNoiser
from wfa_cardinality_estimation_evaluation_framework.common import noisers
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchNoiserBase


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


class GeometricBloomFilter(AnyDistributionBloomFilter):
  """Implement a Geometric Bloom Filter."""

  @classmethod
  def get_sketch_factory(cls, length, probability):
    def f(random_seed):
      return cls(length, probability, random_seed)

    return f

  def __init__(self, length, probability, random_seed=None):
    """Creates a BloomFilter.

    Args:
       length: The length of bit vector for the bloom filter
       probability: p of geometric distribution, p should be small enough
       that geom.cdf(length, probability) won't be 1 in the middle of the
       array so all bits can be used
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    super().__init__(
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.GeometricDistribution(length, probability),
                "geometric")
        ], num_hashes=1, value_functions=[any_sketch.BitwiseOrFunction()]),
        random_seed)


class UniformCountingBloomFilter(any_sketch.AnySketch):
  """Implement a Uniform Counting Bloom Filter."""

  @classmethod
  def get_sketch_factory(cls, length):

    def f(random_seed):
      return cls(length, random_seed)

    return f

  def __init__(self, length, random_seed=None):
    """Creates a Uniform Counting BloomFilter.

    Args:
       length: The length of bit vector for the bloom filter
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    super().__init__(
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.UniformDistribution(length), "uniformcbf")
        ], num_hashes=1, value_functions=[any_sketch.SumFunction()]),
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


class ExponentialBloomFilter(AnyDistributionBloomFilter):
  """Implement an Exponential Bloom Filter."""

  @classmethod
  def get_sketch_factory(cls, length, decay_rate):

    def f(random_seed):
      return cls(length, decay_rate, random_seed)

    return f

  def __init__(self, length, decay_rate, random_seed=None):
    """Creates an ExponentialBloomFilter.

    Args:
       length: The length of bit vector for the bloom filter.
       decay_rate: The decay rate of Exponential distribution.
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    AnyDistributionBloomFilter.__init__(
        self,
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.ExponentialDistribution(length, decay_rate), "exp")
        ], num_hashes=1, value_functions=[any_sketch.BitwiseOrFunction()]),
        random_seed)
    self.decay_rate = decay_rate


class UnionEstimator(EstimatorBase):
  """A class that unions BloomFilters and estimates the combined cardinality."""

  def __init__(self, denoiser=None):
    EstimatorBase.__init__(self)
    if denoiser is None:
      self._denoiser = copy.deepcopy
    else:
      self._denoiser = denoiser

  @classmethod
  def _check_compatibility(cls, sketch_list):
    """Determines if all sketches are compatible."""
    first_sketch = sketch_list[0]
    for cur_sketch in sketch_list[1:]:
      first_sketch.assert_compatible(cur_sketch)

  def union_sketches(self, sketch_list):
    """Exposed for testing."""
    UnionEstimator._check_compatibility(sketch_list)
    sketch_list = self._denoiser(sketch_list)
    union = sketch_list[0]
    for cur_sketch in sketch_list[1:]:
      union.sketch = 1 - (1 - union.sketch) * (1 - cur_sketch.sketch)
    return union

  @classmethod
  def estimate_cardinality(cls, sketch):
    """Estimate the number of elements contained in the BloomFilter."""
    x = np.sum(sketch.sketch)
    k = float(sketch.num_hashes())
    m = float(sketch.max_size())
    if x >= m:
    # When the BF is almost full, the estimate may have large bias or variance.
    # So, later we might change this to x >= z * m where z < 1.
    # We may determine the threshold z based on some theory.
      raise ValueError(
          "The BloomFilter is full. "
          "Please increase the BloomFilter length or use exp/log-BloomFilter.")
    return int(math.fabs(m / k * math.log(1 - x / m)))

  def __call__(self, sketch_list):
    """Does a bit-wise of all sketches and returns a combined cardinality estimate."""
    if not sketch_list:
      return 0
    assert isinstance(sketch_list[0], BloomFilter), "expected a BloomFilter"
    union = self.union_sketches(sketch_list)
    return [UnionEstimator.estimate_cardinality(union)]


class FirstMomentEstimator(EstimatorBase):
  """First moment cardinality estimator for AnyDistributionBloomFilter."""
  # TODO: Refactor this class to break down the methods for each type
  METHOD_UNIFORM = "uniform"
  METHOD_GEO = "geo"
  METHOD_LOG = "log"
  METHOD_EXP = "exp"
  METHOD_ANY = "any"

  def __init__(self, method, denoiser=None, noiser=None, weights=None):
    """Construct an estimator.

    Args:
      method: an estimation method name. One of METHOD_GEO, METHOD_LOG,
        METHOD_EXP, or METHOD_ANY defined in this class.
      denoiser: a callable that conforms to the DenoiserBase. It is used to
        estimate the raw sketch given a sketch with local DP noise.
      noiser: a callable that takes the sum of bits and return the noised sum
        of bits. This is used to add the global DP noise.
      weights: an array of per bucket weights.
    """
    EstimatorBase.__init__(self)

    # The METHOD_ANY requires bucket
    # weights. However, the global noise scenario is supposed to
    # simulate an MPC protocol, which cannot know any bucket
    # weights as this would undo the effects of shuffling.
    if (method == FirstMomentEstimator.METHOD_ANY and noiser is not None):
      raise ValueError(
          "METHOD_ANY and METHOD_GEO are both incompatible with a noiser.")

    if denoiser is None:
      self._denoiser = copy.deepcopy
    else:
      self._denoiser = denoiser

    if noiser is None:
      # This saves some "None" checks in a few functions below
      self._noiser = lambda x: x
    else:
      self._noiser = noiser

    self._weights = weights
    assert method in (
        FirstMomentEstimator.METHOD_UNIFORM,
        FirstMomentEstimator.METHOD_GEO,
        FirstMomentEstimator.METHOD_LOG,
        FirstMomentEstimator.METHOD_EXP,
        FirstMomentEstimator.METHOD_ANY), f"method={method} not supported."
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
    sketch_list = self._denoiser(sketch_list)
    union = sketch_list[0]
    for cur_sketch in sketch_list[1:]:
      union.sketch = 1 - (1 - union.sketch) * (1 - cur_sketch.sketch)
    return union

  @classmethod
  def _estimate_cardinality_uniform(cls, sketch, noiser):
    """Estimate cardinality of a Uniform Bloom Filter."""
    x = noiser(sum(sketch.sketch))
    m = len(sketch.sketch)
    if x >= m or x < 0:
      return float("NaN")
    return -m * math.log(1 - x / m)

  @classmethod
  def _estimate_cardinality_log(cls, sketch, noiser):
    """Estimate cardinality of an Log Bloom Filter."""
    x = int(noiser(sum(sketch.sketch)))
    m = len(sketch.sketch)
    return x / (1 - x / m)

  @classmethod
  def _estimate_cardinality_exp(cls, sketch, noiser):
    """Estimate cardinality of an Exp Bloom Filter a.k.a. Liquid Legions.

    Args:
      sketch: An ExponentialBloomFilter. It should be unnoised or obtained
        after denoising.
    Returns:
      The estimated cardinality of the ADBF.
    """
    a = sketch.decay_rate
    def _expected_num_bits(reach):
      """Expected number of bits activated for cardinality."""
      if reach <= 0:
        return 0
      return 1 - (- special.expi(- a * reach / (np.exp(a) - 1)) +
                  special.expi(- a * np.exp(a) * reach / (np.exp(a) - 1))) / a

    def _clip(x, lower_bound, upper_bound):
      return max(min(x, upper_bound), lower_bound)

    x = int(noiser(sum(sketch.sketch)))
    m = len(sketch.sketch)
    p = _clip(x / m, 0, 1)
    result = invert_monotonic(_expected_num_bits, epsilon=1e-7)(p) * m
    assert result >= 0, "Negative estimate should never happen."
    return result

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

  @classmethod
  def _estimate_cardinality_geo(cls, sketch, noiser):
    """Estimate cardinality of a Bloom Filter with geometric distribution."""
    register_probs = sketch.config.index_specs[0].distribution.register_probs
    n_sum = noiser(sum(sketch.sketch))
    n = n_sum / len(sketch.sketch)

    if(n >= 1):
      return 0
    def first_moment(u):
      return np.sum(1 - np.power(1 - register_probs, u)) - n_sum

    lower_bound = (np.log(1 - n) / np.log(1 - np.mean(register_probs)))
    # In case lower bound is already larger due to noise, we just return
    if first_moment(lower_bound) > 0:
      return lower_bound

    return invert_monotonic(first_moment, lower_bound)(0)

  def __call__(self, sketch_list):
    """Merge all sketches and estimates the cardinality of their union."""
    if not sketch_list:
      return 0
    assert isinstance(sketch_list[0], AnyDistributionBloomFilter), (
        "Expected an AnyDistributionBloomFilter.")
    union = self.union_sketches(sketch_list)

    if self._method == FirstMomentEstimator.METHOD_LOG:
      return [FirstMomentEstimator._estimate_cardinality_log(union, self._noiser)]
    if self._method == FirstMomentEstimator.METHOD_EXP:
      return [FirstMomentEstimator._estimate_cardinality_exp(union, self._noiser)]
    if self._method == FirstMomentEstimator.METHOD_UNIFORM:
      return [FirstMomentEstimator._estimate_cardinality_uniform(union, self._noiser)]
    if self._method == FirstMomentEstimator.METHOD_GEO:
      return [FirstMomentEstimator._estimate_cardinality_geo(
        union, self._noiser)]
    return [FirstMomentEstimator._estimate_cardinality_any(
        union, self._weights)]


class FixedProbabilityBitFlipNoiser(SketchNoiserBase):
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
    SketchNoiserBase.__init__(self)
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


def get_probability_of_flip(epsilon, num_hashes):
  """Get the flipping probability from the privacy epsilon.

  Args:
    epsilon: the differential privacy parameter.
    num_hashes: the number of hash functions used by the bloom filter.

  Returns:
    The flipping probability.
  """
  return 1 / (1 + math.exp(epsilon / num_hashes))


class BlipNoiser(SketchNoiserBase):
  """This class applies "Blip" noise to a BloomFilter.

  This is a common algorithm for making Bloom filters differentially private.
  See [Alaggan et. al 2012] BLIP: Non-interactive Differentially-Private
     Similarity Computation on Bloom filters
  """

  def __init__(self, epsilon, random_state=np.random.RandomState()):
    """Creates a Blip Perturbator.

    Args:
      epsilon: the privacy parameter
      random_state: a numpy.random.RandomState used to draw random numbers
    """
    SketchNoiserBase.__init__(self)
    self._epsilon = epsilon
    self.random_state = random_state

  def __call__(self, bloom_filter):
    """Returns a copy of a BloomFilter with possibly flipped bits.

    Args:
      bloom_filter: The BloomFilter.

    Returns:
      Bit flipped BloomFilter.
    """
    fixed_noiser = FixedProbabilityBitFlipNoiser(
        probability=get_probability_of_flip(self._epsilon,
                                            bloom_filter.num_hashes()),
        random_state=self.random_state)
    return fixed_noiser(bloom_filter)


class DenoiserBase(object):
  """An estimator takes a list of noisy sketches and returns a denoised copy.

  This class should be used before the sketches are sent to the cardinality
  estimator. For example, we calculate the expected register values of an
  AnyDistributionBloomFilter sketch given the observed noisy sketch, which we
  name as a "denoiser".
  """

  def __call__(self, sketch_list):
    """Return a denoised copy of the incoming sketch list."""
    raise NotImplementedError()


class SurrealDenoiser(DenoiserBase):
  """A closed form denoiser for a list of Any Distribution Bloom Filter."""

  def __init__(self, epsilon=None, probability=None):
    """Construct a denoiser.

    Args:
      epsilon: a non-negative differential privacy parameter.
    """
    assert epsilon is not None or probability is not None, (
      "Either epsilon or probability must be given")
    if probability is not None:
      self._probability = probability
    elif epsilon is not None:
      # Currently only support one hash function.
      self._probability = get_probability_of_flip(epsilon, 1)

  def  __call__(self, sketch_list):
    return self._denoise(sketch_list)

  def _denoise(self, sketch_list):
    denoised_sketch_list = []
    for sketch in sketch_list:
      denoised_sketch_list.append(self._denoise_one(sketch))
    return denoised_sketch_list

  def _denoise_one(self, sketch):
    """Denoise a Bloom Filter.

    Args:
      sketch: a noisy Any Distribution Bloom Filter sketch.

    Returns:
      A denoised Any Distribution Bloom Filter.
    """
    assert sketch.num_hashes() == 1, (
        "Currently only support one hash function. "
        "Will extend to multiple hash functions later.")
    denoised_sketch = copy.deepcopy(sketch)
    expected_zeros = (
        - denoised_sketch.sketch * self._probability
        + (1 - denoised_sketch.sketch) * (1 - self._probability))
    denoised_sketch.sketch = 1 - expected_zeros / (
        1 - self._probability - self._probability)
    return denoised_sketch
