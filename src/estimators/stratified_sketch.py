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

import collections
import copy
import functools
import sys
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet

ONE_PLUS = '1+'

class StratifiedSketch(SketchBase):
  """A frequency sketch that contains cardinality sketches per frequency bucket."""

  @classmethod
  def get_sketch_factory(cls):

    def f(random_seed):
      return cls(random_seed=random_seed)

    return f
  def __init__(self, max_freq, sketch_list, random_seed=None):
    """Construct a Stratified sketch.

    Args:
      max_freq: the maximum targeting frequency level. For example, if
        it is set to 3, then the sketches will include frequency=1, 2, and
        frequency >= 3.
      random_seed: This arg exists in order to conform to
        simulator.EstimatorConfig.sketch_factory.
      sketch_list: List of cardinality sketches of the same sketch type.

    """
    SketchBase.__init__(self)
    # A dictionary that contains multiple sketches, which include:
    # (1) sketches with frequency equal to k, where k < max_freq;
    # (2) a sketch with frequency greater than or equal to max_freq;
    # (3) a 1+ reach sketch.
    self.sketches = {}
    self.seed = random_seed
    self.max_freq = max_freq

  def init_from_exact_multi_set(self, exact_multi_set, CardinalitySketch=None):
    """Initialize a Stratified sketch from one ExactMultiSet

    Args:
      exact_multi_set: ExactMultiSet object to use for initialization.
      CardinalitySketch: Class type of cardinality sketches this stratified
        sketch will hold.

    """
    self.sketches[ONE_PLUS] = CardinalitySketch()
    for freq in range(1, self.max_freq+1):
        self.sketches[str(freq)] = CardinalitySketch()

    for id in exact_multi_set.ids():
      id_freq = min(exact_multi_set.frequency(id), self.max_freq)
      self.sketches[str(id_freq)].add(id)
      self.sketches[str(ONE_PLUS)].add(id)

  def init_from_set_generator(self, set_generator, CardinalitySketch=None):
    """Initialize a Stratified sketch from a Set Generator.

    Args:
      set_generator: SetGenerator object to draw ids from for initialization.
      CardinalitySketch: Class type of cardinality sketches this stratified
        sketch will hold.

    """

    exact_multi_set = ExactMultiSet()
    for set in set_generator:
        for item in set:
            exact_multi_set.add(item)
    self.init_from_exact_multi_set(exact_multi_set, CardinalitySketch)

  def assert_compatible(self, other):
    """"Check if the two StratifiedSketch are comparable.

    Args:
      other: the other StratifiedSketch for comparison.

    Raises:
      AssertionError: if the other sketches are not StratifiedSketch, or if
      their random_seed are different, or if the frequency targets are
      different.
    """
    assert isinstance(other, StratifiedSketch), (
        'other is not a StratifiedSketch.')
    assert self.seed == other.seed, (
        'The random seeds are not the same: '
        f'{self.seed} != {other.seed}')
    assert self.max_freq == other.max_freq, (
        'The frequency targets are different: '
        f'{self.max_freq} != {other.max_freq}')

class PairwiseEstimator(object):
  """Merge and estimate two StratifiedSketch."""

  def __init__(self, sketch_operator):
    """Create an estimator for two Stratified sketches.

    Args:
      sketch_operator: an object that have union, intersection, and
        difference methods for two sketches.
    """

    self.sketch_union = sketch_operator.union
    self.sketch_difference = sketch_operator.difference
    self.sketch_intersection = sketch_operator.intersection

  def __call__(self, this, that):
    merged = self.merge_sketches(this, that)
    return merged

  def merge_sketches(self, this, that):
    """Merge two StratifiedSketch.

       Given 2 sketches A and B:
       Merged(k) = (A(k) & B(0)) U (A(k-1) & B(1)) ... U (A(0) & B(k))
         where
         A(k) & B(0) =  A(k) - (A(k) & B(1+))
         B(k) & A(0) =  B(k) - (B(k) & A(1+))
    Args:
      this: one of the two StratifiedSketch to be merged.
      that: the other StratifiedSketch to be merged.

    Returns:
      A merged StratifiedSketch from the input.
    """
    assert isinstance(this, StratifiedSketch)
    this.assert_compatible(that)
    max_freq = this.max_freq
    merged_sketch = copy.deepcopy(this)

    for k in range(max_freq - 1):
      # Calculate A(k) & B(0) = A(k) - (A(k) & B(1+))
      merged = self.sketch_difference(
          this.sketches[str(k + 1)], self.sketch_intersection(
              this.sketches[str(k + 1)], that.sketches[ONE_PLUS]))

      # Calculate A(0) & B(k) = B(k) - (B(k) & A(1+))
      merged = self.sketch_union(
          merged, self.sketch_difference(
              that.sketches[str(k + 1)], self.sketch_intersection(
                  this.sketches[ONE_PLUS], that.sketches[str(k + 1)])))

      # Calculate A(i) & B(k-i)
      for i in range(k):
        merged = self.sketch_union(
            merged, self.sketch_intersection(
                this.sketches[str(i + 1)], that.sketches[str(k - i)]))
      merged_sketch.sketches[str(k + 1)] = merged

    # Calculate Merged(max_freq)
    merged = this.sketches[str(max_freq)]
    rest = that.sketches[ONE_PLUS]
    for k in range(max_freq - 1):
      merged = self.sketch_union(
          merged,
          self.sketch_intersection(this.sketches[str(max_freq - k - 1)],
                                   rest))
      rest = self.sketch_difference(rest, that.sketches[str(k + 1)])
    merged = self.sketch_union(
        merged,
        self.sketch_difference(
            that.sketches[str(max_freq)],
            self.sketch_intersection(
                that.sketches[str(max_freq)],
                this.sketches[ONE_PLUS])))
    merged_sketch.sketches[str(max_freq)] = merged

    # Calculate Merged(1+)
    merged_one_plus = None
    for k in range(max_freq):
      merged_one_plus = self.sketch_union(merged_one_plus,
                                          merged_sketch.sketches[str(k + 1)])
    merged_sketch.sketches[ONE_PLUS] = merged_one_plus

    return merged_sketch

class SequentialEstimator(object):
  """Sequential frequency estimator."""

  def __init__(self, sketch_operator):
    self.pairwise_estimator = PairwiseEstimator(sketch_operator)

  def __call__(self, sketches_list):
    return self.merge_sketches(sketches_list)

  def merge_sketches(self, sketches_list):
    return functools.reduce(self.pairwise_estimator.merge_sketches,
                            sketches_list)

class ExactMultiSetOperation(object):

  @classmethod
  def union(cls, this, that):
    if this is None:
      return copy.deepcopy(that)
    if that is None:
      return copy.deepcopy(this)
    result = copy.deepcopy(this)
    result_key_set = set(result.ids().keys())
    that_key_set = set(that.ids().keys())
    result._ids = {x:1 for x in result_key_set.union(that_key_set)}
    return result

  @classmethod
  def intersection(cls, this, that):
    if this is None or that is None:
      return None
    result = copy.deepcopy(this)
    result_key_set = set(result.ids().keys())
    that_key_set = set(that.ids().keys())
    result._ids = {x:1 for x in result_key_set.intersection(that_key_set)}
    return result

  @classmethod
  def difference(cls, this, that):
    if this is None:
      return None
    if that is None:
      return copy.deepcopy(this)
    result = copy.deepcopy(this)
    result_key_set = set(result.ids().keys())
    that_key_set = set(that.ids().keys())
    result._ids = {x:1 for x in result_key_set.difference(that_key_set)}
    return result
