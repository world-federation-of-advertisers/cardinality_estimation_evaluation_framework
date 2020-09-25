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
"""Sketch operator for Vector-of-Counts."""
import copy
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import PairwiseEstimator


class StratifiedSketchOperator:
  """Sketch operations for supporting frequency dedupe.

  The operators defined in this class are NOT intended to be used as generic
  sketch operators for the Vector-of-Counts (VoC hereafter). They are ONLY
  intended to be used with stratefied_sketch.py for estimation frequency.
  The reason behind this is that the union operator defined here is for VoCs
  whose underlying sets are DISJOINT, and the difference operator is for
  those sets that are fully-overlapping.
  """

  def __init__(self, clip=False, epsilon=np.log(3), clip_threshold=3):
    """Construct a VectorOfCounts sketch operator.

    Args:
      clip: Whether to clip the intersection when merging two VectorOfCounts.
      epsilon: Value of epsilon in differential privacy.
      clip_threshold: Threshold of z-score in clipping. The larger threshold,
        the more chance of clipping.
    """
    self._clip = clip
    self._estimator = PairwiseEstimator(clip=clip, epsilon=epsilon,
                                        clip_threshold=clip_threshold)

  def union(self, this, that):
    """Generate the union sketch of the input sketches.

    Args:
      this: a VectorOfCounts.
      that: a VectorOfCounts which is assumed to be disjoint with this.

    Returns:
      A VectorOfCounts that is the union of this and that.
    """
    if this is None:
      return copy.deepcopy(that)
    if that is None:
      return copy.deepcopy(this)
    result = copy.deepcopy(this)
    result.stats = result.stats + that.stats
    return result

  def intersection(self, this, that):
    """Generate the intersection sketch of the input sketches.

    Args:
      this: a VectorOfCounts.
      that: a VectorOfCounts.

    Returns:
      A VectorOfCounts that is the intersection of the input sketches.
    """
    if this is None or that is None:
      return None

    if self._clip:
      this = self._estimator.clip_empty_vector_of_count(this)
      that = self._estimator.clip_empty_vector_of_count(that)

    union_sketch = self._estimator.merge(this, that)
    result = copy.deepcopy(union_sketch)
    result.stats = this.stats + that.stats - union_sketch.stats
    return result

  def difference(self, this, that):
    """Generate the difference sketch of the input sketches.

    Args:
      this: a VectorOfCounts.
      that: a VectorOfCounts, which is assumed to be a subset of this sketch.

    Returns:
      A VectorOfCounts that is the difference of the input sketches.
    """
    if this is None or that is None:
      return copy.deepcopy(this)
    result = copy.deepcopy(this)
    result.stats = this.stats - that.stats
    return result
