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

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator


class SketchOperator(object):
  """Sketch operators for supporting local DP frequency dedupe."""

  def __init__(self, estimation_method, threshold=1e-6):
    """Construct an AnyDistributionBloomFilter sketch operator.

    Args:
      estimation_method: distribution of bit probabilities in ADBF. That is,
        method to specify in FirstMomentEstimator.
      threshold: a threshold to avoid numerical errors.
    """
    self.estimator = FirstMomentEstimator(method=estimation_method)
    self.threshold = threshold

  def union(self, this, that):
    """Generate the union sketch of the input sketches.

    Args:
      this: an AnyDistributionBloomFilter.
      that: an AnyDistributionBloomFilter with the same config as this. Both
        this and that can be either raw ADBFs or denoised ADBFs.

    Returns:
      An AnyDistributionBloomFilter representing the union of this and that.
    """
    if this is None:
      return copy.deepcopy(that)
    this.assert_compatible(that)
    result = copy.deepcopy(this)
    result.sketch = 1 - (1 - this.sketch) * (1 - that.sketch)
    return result

  @classmethod
  def _get_register_probs(cls, adbf):
    return adbf.config.index_specs[0].distribution.register_probs

  @classmethod
  def _predict_registers(cls, register_probs, cardinality):
    """A common function used in the intersection and difference operators."""
    return 1 - np.power(1 - register_probs, cardinality)

  def _obtain_two_way_venn_diagram(self, this, that):
    self.this_cardinality = self.estimator([this])[0]
    self.that_cardinality = self.estimator([that])[0]
    union_cardinality = self.estimator([this, that])[0]
    self.intersection_cardinality = max(
        self.this_cardinality + self.that_cardinality - union_cardinality, 0)

  def intersection(self, this, that):
    raise NotImplementedError()

  def difference(self, this, that):
    raise NotImplementedError()


class BayesianApproximationSketchOperator(SketchOperator):
  """Sketch operators based on the Bayesian method."""

  def __init__(self, estimation_method, threshold=1e-6):
    """Construct an AnyDistributionBloomFilter sketch operator.

    Args:
      estimation_method: distribution of bit probabilities in ADBF. That is,
        method to specify in FirstMomentEstimator.
      threshold: a threshold to avoid numerical errors.
    """
    SketchOperator.__init__(self, estimation_method, threshold)

  def intersection(self, this, that):
    """Generate the intersection sketch of the input sketches.

    Args:
      this: an AnyDistributionBloomFilter.
      that: an AnyDistributionBloomFilter with the same config as this. Both
        this and that can be either raw ADBFs or denoised ADBFs.

    Returns:
      An AnyDistributionBloomFilter representing the intersection of this and
        that.
    """
    if this is None or that is None:
      return None
    this.assert_compatible(that)
    result = copy.deepcopy(this)
    self._obtain_two_way_venn_diagram(this, that)
    register_probs = self._get_register_probs(this)
    hc11 = self._predict_registers(
        register_probs=register_probs,
        cardinality=self.intersection_cardinality)
    hc10 = self._predict_registers(
        register_probs=register_probs,
        cardinality=self.this_cardinality - self.intersection_cardinality)
    hc01 = self._predict_registers(
        register_probs=register_probs,
        cardinality=self.that_cardinality - self.intersection_cardinality)
    y = hc11 / np.maximum(self.threshold, hc10 * hc01 * (1 - hc11) + hc11)
    result.sketch = this.sketch * that.sketch * y
    return result

  def difference(self, this, that):
    """Generate the difference sketch of the input sketches.

    Args:
      this: an AnyDistributionBloomFilter.
      that: an AnyDistributionBloomFilter with the same config as this. Both
        this and that can be either raw ADBFs or denoised ADBFs.

    Returns:
      An AnyDistributionBloomFilter representing the difference of this - that.
    """
    if this is None:
      return None
    if that is None:
      return this
    this.assert_compatible(that)
    result = copy.deepcopy(this)
    self._obtain_two_way_venn_diagram(this, that)
    register_probs = self._get_register_probs(this)
    hc11 = self._predict_registers(
        register_probs=register_probs,
        cardinality=self.intersection_cardinality)
    hc10 = self._predict_registers(
        register_probs=register_probs,
        cardinality=self.this_cardinality - self.intersection_cardinality)
    hc01 = self._predict_registers(
        register_probs=register_probs,
        cardinality=self.that_cardinality - self.intersection_cardinality)
    denominator = np.maximum(self.threshold, hc10 * hc01 * (1 - hc11) + hc11)
    numerator = (hc10 * hc01 * hc11 + hc10 * (1 - hc01) * hc11
                 + hc10 * hc01 * (1 - hc11))
    y = numerator / denominator
    result.sketch = (this.sketch * (1 - that.sketch)
                     + this.sketch * that.sketch * y)
    return result


class ExpectationApproximationSketchOperator(SketchOperator):
  """Sketch operators based on the Expectation method."""

  def __init__(self, estimation_method, threshold=1e-6):
    """Construct an AnyDistributionBloomFilter sketch operator.

    Args:
      estimation_method: distribution of bit probabilities in ADBF. That is,
        method to specify in FirstMomentEstimator.
      threshold: a threshold to avoid numerical errors.
    """
    SketchOperator.__init__(self, estimation_method, threshold)

  def intersection(self, this, that):
    """Generate the intersection sketch of the input sketches.

    Args:
      this: an AnyDistributionBloomFilter.
      that: an AnyDistributionBloomFilter with the same config as this. Both
        this and that can be either raw ADBFs or denoised ADBFs.

    Returns:
      An AnyDistributionBloomFilter representing the intersection of this and
        that.
    """
    if this is None or that is None:
      return None
    this.assert_compatible(that)
    result = copy.deepcopy(this)
    self._obtain_two_way_venn_diagram(this, that)
    register_probs = self._get_register_probs(this)
    x = max(np.sum(register_probs * this.sketch * that.sketch),
            self.threshold, np.min(register_probs))
    y = SketchOperator._predict_registers(
        register_probs=register_probs / x,
        cardinality=self.intersection_cardinality)
    result.sketch = this.sketch * that.sketch * y
    return result

  def difference(self, this, that):
    """Generate the difference sketch of the input sketches.

    Args:
      this: an AnyDistributionBloomFilter.
      that: an AnyDistributionBloomFilter with the same config as this. Both
        this and that can be either raw ADBFs or denoised ADBFs.

    Returns:
      An AnyDistributionBloomFilter representing the difference of this - that.
    """
    if this is None:
      return None
    this.assert_compatible(that)
    result = copy.deepcopy(this)
    self._obtain_two_way_venn_diagram(this, that)
    register_probs = self._get_register_probs(this)
    x = max(np.sum(register_probs * this.sketch),
            self.threshold, np.min(register_probs))
    s = (self.this_cardinality - self.intersection_cardinality
         - np.dot(this.sketch, 1 - that.sketch))
    s = max(0, min(self.this_cardinality - self.intersection_cardinality, s))
    y = SketchOperator._predict_registers(
        register_probs=register_probs / x, cardinality=s)
    result.sketch = (this.sketch * (1 - that.sketch)
                     + this.sketch * that.sketch * y)
    return result

