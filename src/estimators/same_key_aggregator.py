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
"""Same key aggregator for frequency estimation."""

import copy
import functools
import inspect
import numpy as np
import warnings

from wfa_cardinality_estimation_evaluation_framework.estimators import any_sketch
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import UniqueKeyFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filter_sketch_operators import SketchOperator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import ExponentialBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import DiscreteGaussianEstimateNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import GaussianEstimateNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import GeometricEstimateNoiser


class ExponentialSameKeyAggregator(SketchBase):
  """Implement a Same Key Aggregator in Exponential bloom filter."""

  @classmethod
  def get_sketch_factory(cls, length, decay_rate):

    def f(random_seed):
      return cls(length, decay_rate, random_seed)

    return f

  def __init__(self, length, decay_rate, random_seed):
    """Creates an ExponentialSameKeyAggregator.

    An ExponentialSameKeyAggregator includes three components:
    1. An ExponentialBloomFilter for estimating the reach.
    2. An AnySketch to track the unique key in each register.
    3. Another AnySketch to track the frequency of each effective key.

    Args:
       length: The length of bit vector for the Exponential bloom filter.
       decay_rate: The decay rate of Exponential distribution.
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    self.length = length
    self.decay_rate = decay_rate
    self.exponential_bloom_filter = ExponentialBloomFilter(
        length=length, decay_rate=decay_rate, random_seed=random_seed)
    self.unique_key_tracker = any_sketch.AnySketch(
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.ExponentialDistribution(length, decay_rate), 'exp')
        ], num_hashes=1, value_functions=[any_sketch.UniqueKeyFunction()]),
        random_seed)
    self.frequency_count_tracker = any_sketch.AnySketch(
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.ExponentialDistribution(length, decay_rate), 'exp')
        ], num_hashes=1, value_functions=[any_sketch.SumFunction()]),
        random_seed)

  def add(self, x):
    self.exponential_bloom_filter.add(x)
    self.frequency_count_tracker.add(x)
    # The unique_key_sketch needs to be updated in a special way
    indexes = self.unique_key_tracker.get_indexes(x)
    unique_key = UniqueKeyFunction()
    for index in indexes:
      self.unique_key_tracker.sketch[index] = unique_key(
          self.unique_key_tracker.sketch[index],
          UniqueKeyFunction.get_value_from_id(x))

  def assert_compatible(self, other):
    """"Check if the two ExponentialSameKeyAggregator are comparable."""
    assert isinstance(other, ExponentialSameKeyAggregator), (
        'Other is not a ExponentialSameKeyAggregator.')
    assert self.length == other.length, (
        'The sketch lengths are different: '
        f'{self.length} != {other.length}')
    assert self.decay_rate == other.decay_rate, (
        'The decay rates are different: '
        f'{self.decay_rate} != {other.decay_rate}')


class StandardizedHistogramEstimator(EstimatorBase):
  """Frequency estimator from ExponentialSameKeyAggregator.

    Algorithm Description
    (following the argument names in __init__ and __call__):
    Given any sketch_list which includes the ExponentialSameKeyAggregator from
    a set of publishers, we estimate the frequency histogram among all the
    publishers in the following steps.
    Step 1. Estimate the one_plus_reach, which is 1+ union reach of all pubs.
    Step 2. Estimate the active_key_histogram, which is the histogram of
      the frequency for each active key, i.e., each id that is the unique
      reached id in its bucket.
    Step 3. Noise one_plus_reach by
      noiser_class(reach_epsilon, **reach_noiser_kwargs).
      And noise active_key_histogram by
      noiser_class(frequency_epsilon, **frequency_noiser_kwargs).
    Step 4. Return noised_one_plus_reach * noised_active_key_histogram /
      sum(noised_active_key_histogram).
  """

  def __init__(self,
               max_freq=10,
               reach_noiser_class=GeometricEstimateNoiser,
               frequency_noiser_class=GeometricEstimateNoiser,
               reach_epsilon=1.,
               frequency_epsilon=1.,
               reach_delta=0.,
               frequency_delta=0.,
               reach_noiser_kwargs={},
               frequency_noiser_kwargs={}):
    """Initiate a StandardizedHistogramEstimator.

    To understand the usage of the arguments here, please read the
    "Algorithm Description" in the docstring of the whole class.

    Args:
      max_freq: the maximum targeting frequency level. For example, if it is set
        to 3, then the sketches will include frequency=1, 2, 3+ (frequency >=
        3). Note: we have to set a max_freq; privacy cannot be guaranteed if
        there's no max_freq.
      noiser_class: a class of noiser indicating the distribution of noise.
      reach_epsilon: DP-epsilon for the noise on the one_plus_reach.
      frequency_epsilon: DP-epsilon for the noise on active_key_histogram.
      reach_noiser_kwargs: a dictionary of arguments other than epsilon in the
        noiser for one_plus_reach. It can be specified in the following ways.
        1. When noiser_class=LaplaceEstimateNoiser or GeometricEstimateNoiser,
        reach_noiser_kwargs can be an empty dictionary, or
        {'random_state': <a np.random.RandomState>}.
        2. When noiser_class=GaussianEstimateNoiser or DiscreteGaussianEstimateNoiser,
        one must specify a delta argument for the (epsilon, delta)-DP.
        Thus, reach_noiser_kwargs can be {'delta': <a float between 0 and 1>},
        or {'delta': <a float between 0 and 1>,
         'random_state': <a np.random.RandomState>}.
        See wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers
        for other possible ways to specify reach_noiser_para.
      frequency_noiser_kwargs:  a dictionary of arguments other than epsilon in
        the noiser for active_key_histogram. It's specified in the same way with
        reach_noiser_kwargs.

        To further illustrate how to build the noisers with the above arguments,
        see the following example. With
        noiser_class=DiscreteGaussianEstimateNoiser,
        epsilon=1, epsilon_split=0.3,
        reach_noiser_kwargs={'delta': 1e-5, 'random_state': np.random.RandomState(1)},
        frequency_noiser_kwargs={'delta': 1e-6, 'random_state': np.random.RandomState(2)},
        the estimated one_plus_reach is noised by a discrete Gaussian
        distribution guaranteeing a (0.3, 1e-5)-DP with a random seed 1.
        And with a random seed 2, the array active_key_histogram is
        noised by i.i.d. discrete Gaussian distribution guaranteeing a
        (0.3, 1e-5)-DP on each count.
    """
    self.max_freq = max_freq
    self.one_plus_reach_noiser = None
    self.histogram_noiser = None


  def __call__(self, sketch_list):
    ska = StandardizedHistogramEstimator.merge_sketch_list(sketch_list)
    return self.estimate_cardinality(ska)

  def get_noiser():
    name = inspect.getfullargspec(noiser_class.__init__)

  @classmethod
  def get_total_privacy_budget(
      cls, reach_noiser_class, reach_epsilon, reach_delta,
      frequency_noiser_class, frequency_epsilon, frequency_delta, max_freq):
    return (reach_epsilon + frequency_epsilon * max_freq,
            reach_delta + frequency_delta * max_freq)
    # Currently, use the naive composition theorem.
    # TODO(jiayu): Work with Pasin and Matthew to find the tight composition.
    # Probably need to use PLD.

  @classmethod
  def merge_two_exponential_bloom_filters(cls, this, that):
    sketch_operator = SketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_EXP)
    return sketch_operator.union(this, that)

  @classmethod
  def merge_two_unique_key_trackers(cls, this, that):
    result = copy.deepcopy(this)
    unique_key_function = UniqueKeyFunction()
    result.sketch = np.array(
        [unique_key_function(x, y) for x, y in zip(this.sketch, that.sketch)])
    return result

  @classmethod
  def merge_two_frequency_count_trackers(cls, this, that):
    result = copy.deepcopy(this)
    result.sketch = this.sketch + that.sketch
    return result

  @classmethod
  def merge_two_sketches(cls, this, that):
    assert isinstance(this, ExponentialSameKeyAggregator), (
        'This is not a ExponentialSameKeyAggregator.')
    result = copy.deepcopy(this)
    result.exponential_bloom_filter = StandardizedHistogramEstimator.merge_two_exponential_bloom_filters(
        this.exponential_bloom_filter, that.exponential_bloom_filter)
    result.unique_key_tracker = StandardizedHistogramEstimator.merge_two_unique_key_trackers(
        this.unique_key_tracker, that.unique_key_tracker)
    result.frequency_count_tracker = StandardizedHistogramEstimator.merge_two_frequency_count_trackers(
        this.frequency_count_tracker, that.frequency_count_tracker)
    return result

  @classmethod
  def merge_sketch_list(cls, sketch_list):
    return functools.reduce(StandardizedHistogramEstimator.merge_two_sketches,
                            sketch_list)

  def estimate_one_plus_reach(self, exponential_same_key_aggregator):
    estimator = FirstMomentEstimator(
        noiser=self.one_plus_reach_noiser,
        method=FirstMomentEstimator.METHOD_EXP)
    return estimator(
        [exponential_same_key_aggregator.exponential_bloom_filter])[0]

  def estimate_histogram_from_effective_keys(
      self, exponential_same_key_aggregator):
    """Obtain the frequency distribution among effective keys.

    Args:
      exponential_same_key_aggregator: an ExponentialSameKeyAggregator.

    Returns:
      An array with any ith element indicating the number of effective keys
      with frequency = (i + 1); the last element indicating the number of
      effective keys with frequency >= self.max_freq. Each element of this array
      has been independently noised by self.histogram_noiser.
    """
    is_effective_register = np.isin(
        exponential_same_key_aggregator.unique_key_tracker.sketch,
        (UniqueKeyFunction.FLAG_EMPTY_REGISTER,
         UniqueKeyFunction.FLAG_COLLIDED_REGISTER),
        invert=True)
    freq_effective_keys = (
        exponential_same_key_aggregator.frequency_count_tracker.sketch
        [is_effective_register])
    if self.max_freq is not None:
      freq_effective_keys[freq_effective_keys > self.max_freq] = self.max_freq
    raw_histogram_array_from_effective_keys = np.bincount(
        freq_effective_keys, minlength=self.max_freq + 1)[1:]
    if self.histogram_noiser is None:
      return raw_histogram_array_from_effective_keys
    return self.histogram_noiser(raw_histogram_array_from_effective_keys)

  @classmethod
  def standardize_histogram(cls, histogram, total):
    """Scales a histogram (array) so that it sums up to a given total."""
    return histogram / sum(histogram) * total

  def estimate_cardinality(self, exponential_same_key_aggregator):
    """Estimate_cardinality of 1+, 2+, ..., N+ reach, from a SameKeyAggregator.

    Args:
      exponential_same_key_aggregator: an ExponentialSameKeyAggregator.

    Returns:
      A list with the ith element being the estimated (i+1)+ reach, i.e.,
      the number of audience who have been exposed to the ads for at least
      (i+1) times. This has the same format as the output of
      stratified_sketch.SequentialEstimator.
    """
    one_plus_reach = self.estimate_one_plus_reach(
        exponential_same_key_aggregator)
    hist = self.estimate_histogram_from_effective_keys(
        exponential_same_key_aggregator)
    standardized_hist = StandardizedHistogramEstimator.standardize_histogram(
        hist, one_plus_reach)
    return list(reversed(np.cumsum(list(reversed(standardized_hist)))))
