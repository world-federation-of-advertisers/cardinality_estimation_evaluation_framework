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


class SketchOperator:
  """Sketch operations for supporting frequency dedupe.

  The operator defined in this class is NOT intended to be used as a generic
  sketch operator for the Vector-of-Counts (VoC hereafter). It is ONLY intended
  to be worked with the stratefied_sketch.py for estimating the frequency. The
  reason behind is that the union operators defined here is for VoCs whose
  underlying sets are DISJOINT, and the difference operator is for those sets
  that are fully-overlapping.
  """

  def __init__(self, estimator):
    """Construct a VectorOfCounts sketch operator.

    Args:
      estimator: an instance generated from the derived class of
        EstimatorBase, which takes in a list of VectorOfCounts sketches and
        return the cardinality. For example, SequentialEstimator.
    """
    self._estimator = estimator

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
    result = copy.deepcopy(this)
    this_cardinality = this.cardinality()
    that_cardinality = that.cardinality()
    union_cardinality = self._estimator([this, that])[0]
    intersection_cardinality = (this_cardinality + that_cardinality
                                - union_cardinality)
    result.stats = intersection_cardinality * (this.stats + that.stats) / (
        this_cardinality + that_cardinality)
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
      return this
    result = copy.deepcopy(this)
    result.stats = this.stats - that.stats
    return result
