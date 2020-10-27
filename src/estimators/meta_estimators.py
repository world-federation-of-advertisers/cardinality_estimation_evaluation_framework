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
"""Implementations of estimators which count Bloom Filter indices."""
import copy

import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
import wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters as bf
import wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts as voc


class MetaEstimatorBase(EstimatorBase):
  """Defines the protocol for meta-estimation.

  Meta-estimation uses a cardinality estimator in order to estimate bucket
  indices in a Bloom Filter set to 1. If the cardinality estimator in question
  is an exact set, this estimator is equivalent to the Bloom Filter base
  estimator.
  """

  def __init__(self,
               meta_sketch_factory,
               meta_sketch_estimator,
               adbf_estimator):
    """Initializes a MetaEstimator using the given FirstMomentEstimator method.

    Args:
      meta_sketch_factory: a sketch factory method used to estimate Bloom Filter
        buckets.
      meta_sketch_estimator: an estimator to estimate the union cardinality
        given a list of meta sketches.
      adbf_estimator: an estimator to estimate the cardinality of a list of
        AnyDistributionBloomFilter sketches.
    """
    super().__init__()
    self.adbf_estimator = adbf_estimator
    self.meta_sketch_factory = meta_sketch_factory
    self.meta_sketch_estimator = meta_sketch_estimator

  @classmethod
  def _check_compatibility(cls, sketch_list):
    """Determines if all sketches are compatible."""
    for sketch in sketch_list:
      assert isinstance(sketch, bf.AnyDistributionBloomFilter), (
          'MetaEstimator must be used with an AnyDistributionBloomFilter.')

    first_sketch = sketch_list[0]

    for cur_sketch in sketch_list[1:]:
      first_sketch.assert_compatible(cur_sketch)

  def __call__(self, sketch_list):
    """Transform AnyDistributionBloomFilters into meta sketch and estimate union."""
    MetaEstimatorBase._check_compatibility(sketch_list)
    if not sketch_list:
      return 0

    meta_sketches = self._transform_adbf_into_meta_sketches(sketch_list)
    num_ones = self._estimate_num_ones(meta_sketches)
    adbf_sketch = MetaEstimatorBase._construct_fake_adbf(
        num_ones, sketch_list[0])
    return self.adbf_estimator([adbf_sketch])

  def _transform_adbf_into_meta_sketches(self, sketch_list):
    """Transform AnyDistributionBloomFilters into meta sketch.

    Args:
      sketch_list: a list of AnyDistributionBloomFilter sketches.

    Returns:
      A list of meta sketches.
    """
    if not sketch_list:
      return []

    random_seed = sketch_list[0].random_seed

    meta_sketches = []
    for sketch in sketch_list:
      meta_sketch = self.meta_sketch_factory(random_seed)

      # Get nonzero indices in sketch and add those indices to the meta_sketch.
      nonzero_indices = np.flatnonzero(sketch.sketch)
      meta_sketch.add_ids(nonzero_indices)

      meta_sketches.append(meta_sketch)

    return meta_sketches

  def _estimate_num_ones(self, meta_sketches):
    """Estimate the number of one registers from the meta sketches.

    Args:
      meta_sketches: a list of meta sketches.

    Returns:
      The estimated number of one registers.
    """
    # Estimate the number of 1's from the meta sketches.
    return int(self.meta_sketch_estimator(meta_sketches)[0])

  @classmethod
  def _construct_fake_adbf(cls, num_ones, template_adbf):
    """Create a fake AnyDistributionBloomFilter.

    We create a fake ADBF which can be used to estimate the cardinality.
    The first num_ones of register will be set to 1, and 0 otherwise.
    As such, we can reuse the FirstMomentEstimator.
    Note that it is only compatible with estimators that doesn't rely on the
    per bucket probability.
    want to be able to reuse the FirstMomentEstimator code here
    which requires that we input a sketch. In order to handle this,
    we will fill a new sketch with union_size number of ones in the
    first n indices. This may break if we use estimators which care
    about the specific bucket indices. This is not currently supported
    behavior.

    Args:
      num_ones: the number of one registers.
      template_adbf: a template AnyDistributionBloomFilter. The created sketch
        will be used to estimate the union cardinality.

    Returns:
      An AnyDistributionBloomFilter, which has the same setting as the
      template_adbf sketch.
    """
    new_sketch = copy.deepcopy(template_adbf)
    new_sketch.sketch[:num_ones] = 1
    new_sketch.sketch[num_ones:] = 0
    return new_sketch


class MetaVoCEstimator(MetaEstimatorBase):
  """A Meta-Estimator using Vectors of Counts and the SequentialEstimator."""

  def __init__(self,
               num_buckets,
               adbf_estimator,
               clip=False,
               epsilon=np.log(3),
               clip_threshold=3):
    """Initializes this MetaVoCEstimator.

    Args:
      num_buckets: Number of buckets in each Meta-VoC
      adbf_estimator: an estimator to estimate the cardinality of a list of
        AnyDistributionBloomFilter sketches.
      clip: Whether to clip the intersection when merging two VectorOfCounts.
      epsilon: Value of epsilon in differential privacy.
      clip_threshold: Threshold of z-score in clipping. The larger threshold,
        the more chance of clipping.
    """
    assert num_buckets > 0, 'MetaVoCEstimator must have at least one bucket.'
    super().__init__(
        voc.VectorOfCounts.get_sketch_factory(num_buckets),
        voc.SequentialEstimator(
            clip=clip, epsilon=epsilon, clip_threshold=clip_threshold),
        adbf_estimator,
    )