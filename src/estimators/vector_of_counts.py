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

"""Vector of Counts cardinality estimator.

This is a new, open source method that has lots of nice properties.  Namely,
because you have to deduplicate users before constructing the Sketch, it is
not, strictly-speaking, a "Cardinality Estimator" as outlined in the paper
titled "Cardinality Estimators do not Preserve Privacy":
https://arxiv.org/pdf/1808.05879.pdf.  So you can construct a vector with
differential privacy, and send it to a third party with a differential privacy
guarantee, but also not worry about losing accuracy as many of the structures
are aggregated together to get an estimate for the cardinality.
"""

import copy

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common.hash_function import HashFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchNoiserBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase


class IdentityNoiser(SketchNoiserBase):
  """Does not add noise to a VectorOfCounts."""

  def __call__(self, sketch):
    """Return an identical copy of the incoming sketch."""
    return copy.deepcopy(sketch)


class LaplaceNoiser(SketchNoiserBase):
  """This class adds noise to a VectorOfCounts."""

  def __init__(self, epsilon=np.log(3), random_state=np.random.RandomState()):
    """Creates a VectorOfCountsNoiser which can add noise to a VectorOfCounts.

    Args:
       epsilon: the privacy parameter.
       random_state: a numpy.random.RandomState used to draw random numbers.
    """
    self.epsilon = epsilon
    self.random_state = random_state

  def __call__(self, sketch):
    """Returns a copy of VectorOfCounts with noise.

    Args:
      sketch: a VectorOfCounts object.

    Returns:
      A new VectorOfCounts object with noise added to the sketch.
    """
    noised_sketch = copy.deepcopy(sketch)
    noise = self.random_state.laplace(loc=0, scale=1.0/self.epsilon,
                                      size=noised_sketch.num_buckets)
    noised_sketch.stats = noised_sketch.stats + noise
    return noised_sketch


class VectorOfCounts(SketchBase):
  """A vector of counts sketch."""

  @classmethod
  def get_sketch_factory(cls, num_buckets, *args, **kwargs):

    def f(random_seed):
      return cls(num_buckets, random_seed)

    return f

  def __init__(self, num_buckets, random_seed):
    """Creates a vector of counts sketch.

    Args:
      num_buckets: the number of buckets of the VectorOfCounts.
      random_seed: a random seed for the hash function.
    """
    SketchBase.__init__(self)
    self._num_buckets = num_buckets
    self.stats = np.zeros(num_buckets)
    self.hash_function = HashFunction(random_seed, num_buckets)
    self._vectorized_hash_function = np.vectorize(self.hash_function)
    self._ids_added = False

  @property
  def num_buckets(self):
    return self._num_buckets

  def add_ids(self, ids):
    """Add IDs to the sketch.

    This method should be called only once. Otherwise, will raise error.
    The reason is that a vector of counts does not support sequentially adding
    ids without additional layers. Let's imagine the following scenario:
    voc = VectorOfCounts(very_large_num_buckets_so_there_is_no_hash_collision)
    voc.add_ids([1, 2])
    voc.add_ids([1, 3])
    Ideally, we want voc.estimate_cardinality() == 3. However, this may be hard,
    unless the vector of counts has a way to tell what ids have been added to
    the bucket so far.

    Args:
      ids: a list of raw ids.

    Returns:
      self.

    Raises:
      AssertionError: If this method is called a second time.
    """
    assert not self._ids_added, 'Can only add ids to Vector of Counts once.'
    hashed_ids = self._vectorized_hash_function(ids)
    self.stats = np.bincount(hashed_ids, minlength=self.num_buckets)
    self._ids_added = True
    return self

  def cardinality(self):
    return np.sum(self.stats)


class PairwiseEstimator(EstimatorBase):
  """A cardinality estimator for two VectorOfCounts."""

  def __init__(self, clip=False, epsilon=np.log(3), clip_threshold=3):
    """Initializes the instance.

    Args:
      clip: Whether to clip the intersection when merging two VectorOfCounts.
      epsilon: Value of epsilon in differential privacy.
      clip_threshold: Threshold of z-score in clipping. The larger threshold,
        the more chance of clipping.
    """
    EstimatorBase.__init__(self)
    self.clip = clip
    self.epsilon = epsilon
    self.clip_threshold = clip_threshold

  @classmethod
  def assert_compatible(cls, this, that):
    """"Check if the two VectorOfCounts are comparable for dedupe.

    Args:
      this: one of the two VectorOfCounts for dedupe.
      that: the other VectorOfCounts for dedupe.

    Raises:
      AssertionError: if the input sketches are not VectorOfCounts, or if their
      lengths are different.
    """
    assert isinstance(this, VectorOfCounts), 'this is not a VectorOfCounts.'
    assert isinstance(that, VectorOfCounts), 'that is not a VectorOfCounts.'
    assert this.hash_function == that.hash_function, (
        'The input VectorOfCounts do not have the same hash function.')
    assert this.num_buckets == that.num_buckets, 'VectorOfCounts size mismatch'

  def __call__(self, sketch_list):
    """Estimates the cardinality of the union of two VectorOfCounts."""
    assert len(sketch_list) == 2
    return [self._union(sketch_list[0], sketch_list[1])]

  @classmethod
  def _intersection(cls, this, that, this_cardinality=None,
                    that_cardinality=None):
    """"Estimate the intersection of two VectorOfCounts.

    Args:
      this: one of the two VectorOfCounts for dedupe.
      that: the other VectorOfCounts for dedupe.
      this_cardinality: the cardinality of this VectorOfCounts.
      that_cardinality: the cardinality of that VectorOfCounts.

    Returns:
      The estimated intersection of two input VectorOfCounts.
    """
    PairwiseEstimator.assert_compatible(this, that)
    this_cardinality = this.cardinality() or this_cardinality
    that_cardinality = that.cardinality() or that_cardinality
    bucket_sizes_this = this_cardinality / this.num_buckets
    bucket_sizes_that = that_cardinality / that.num_buckets
    return np.dot(this.stats - bucket_sizes_this,
                  that.stats - bucket_sizes_that)

  @classmethod
  def _union(cls, this, that):
    this_cardinality = this.cardinality()
    that_cardinality = that.cardinality()
    intersection_cardinality = PairwiseEstimator._intersection(
        this, that, this_cardinality, that_cardinality)
    return this_cardinality + that_cardinality - intersection_cardinality

  def _get_std_of_intersection(self, intersection_cardinality, this, that):
    variance = (this.cardinality() * that.cardinality() +
                np.square(intersection_cardinality)) / this.num_buckets
    variance += this.num_buckets * 4 / self.epsilon ** 4
    variance += (
        this.cardinality() + that.cardinality()) * 2 / self.epsilon ** 2
    return np.sqrt(variance)

  def evaluate_closeness_to_a_value(
      self, intersection_cardinality, value_to_compare_with, this, that):
    """Evaluate if the intersection is close to a value via hypothesis test.

    Args:
      intersection_cardinality: An estimate of intersection.
      value_to_compare_with: We test the hypothesis H0:
        intersection_cardinality = value_to_compare.
      this: one of the two VectorOfCounts for dedupe.
      that: the other VectorOfCounts for dedupe.

    Returns:
      A Z-score describing how close the intersection estimate is close to
        the value.
    """
    return (
        intersection_cardinality - value_to_compare_with
    ) / self._get_std_of_intersection(value_to_compare_with, this, that)

  def has_zero_intersection(self, intersection_cardinality, this, that):
    value_to_compare_with = 0
    z_score = self.evaluate_closeness_to_a_value(
        intersection_cardinality, value_to_compare_with, this, that)
    return z_score < self.clip_threshold

  def has_full_intersection(self, intersection_cardinality, this, that):
    value_to_compare_with = min(this.cardinality(), that.cardinality())
    z_score = self.evaluate_closeness_to_a_value(
        intersection_cardinality, value_to_compare_with, this, that)
    return z_score > - self.clip_threshold

  def merge(self, this, that):
    """Merge two VectorOfCounts.

    Args:
      this: one of the two VectorOfCounts to be merged.
      that: the other VectorOfCounts to be merged.

    Returns:
      A VectorOfCounts object, which is merged.
    """
    PairwiseEstimator.assert_compatible(this, that)
    this_cardinality = this.cardinality()
    that_cardinality = that.cardinality()
    intersection_cardinality = PairwiseEstimator._intersection(
        this, that, this_cardinality, that_cardinality)
    merged = copy.deepcopy(this)
    if self.clip:
      if self.has_zero_intersection(intersection_cardinality, this, that):
        merged.stats = this.stats + that.stats
        return merged
      if self.has_full_intersection(intersection_cardinality, this, that):
        return merged

    if this_cardinality + that_cardinality == 0:
      # It is possible that the sum of cardinalities is 0 and the
      # cardinalities themselves are not 0 under the local DP cases.
      # So need to check this to avoid division by zero.
      # If the sum is zero, will distribute the counts evenly for all the
      # buckets.
      share = np.ones_like(merged.stats) * (
          intersection_cardinality / merged.num_buckets)
      merged.stats = this.stats + that.stats - share
      return merged

    share = intersection_cardinality * (this.stats + that.stats) / (
        this_cardinality + that_cardinality)
    merged.stats = this.stats + that.stats - share
    return merged

  def _get_std_of_sketch_sum(self, sketch):
    return  np.sqrt(sketch.num_buckets * 2) / self.epsilon

  def clip_empty_vector_of_count(self, sketch):
    assert isinstance(sketch, VectorOfCounts), 'Not a VectorOfCounts.'
    z_score = np.sum(sketch.stats) / self._get_std_of_sketch_sum(sketch)
    if z_score < self.clip_threshold:
      sketch.stats = np.zeros(sketch.num_buckets)
    return sketch


class SequentialEstimator(EstimatorBase):
  """An estimator by merging VectorOfCounts by the given order."""

  def __init__(self, clip=False, epsilon=np.log(3), clip_threshold=3):
    """Initializes the instance.

    Args:
      clip: Whether to clip the intersection when merging two VectorOfCounts.
      epsilon: Value of epsilon in differential privacy.
      clip_threshold: Threshold of z-score in clipping. The larger threshold,
        the more chance of clipping.
    """
    EstimatorBase.__init__(self)
    self.clip = clip
    self.epsilon = epsilon
    self.clip_threshold = clip_threshold

  def __call__(self, sketch_list):
    """Estimates the cardinality of the union of a list of VectorOfCounts."""
    return [self._estimate_cardinality(sketch_list)]

  def _estimate_cardinality(self, sketch_list):
    """Merge a list of VectorOfCounts and estimate the cardinality.

    Args:
      sketch_list: a list of VectorOfCounts sketches.

    Returns:
      The estimated cardinality of the merged sketches.
    """
    pairwise_estimator = PairwiseEstimator(
        clip=self.clip, epsilon=self.epsilon,
        clip_threshold=self.clip_threshold)
    if self.clip:
      sketch_list = [pairwise_estimator.clip_empty_vector_of_count(sketch)
                     for sketch in sketch_list]

    current = sketch_list[0]
    for sketch in sketch_list[1:]:
      current = pairwise_estimator.merge(current, sketch)
    return current.cardinality()
