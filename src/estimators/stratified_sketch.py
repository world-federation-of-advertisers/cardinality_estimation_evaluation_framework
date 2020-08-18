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

import copy
import functools
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet

ONE_PLUS = '1+'


class StratifiedSketch(object):
  """A frequency sketch that contains cardinality sketches per frequency bucket."""

  @classmethod
  def get_sketch_factory(cls):

    def f(random_seed):
      return cls(random_seed=random_seed)

    return f

  def __init__(self, max_freq, cardinality_sketch_factory, random_seed):
    """Construct a Stratified sketch.

    Args:
      max_freq: the maximum targeting frequency level. For example, if it is set
        to 3, then the sketches will include frequency=1, 2, 3+ (frequency >=
        3).
      random_seed: This arg exists in order to conform to
        simulator.EstimatorConfig.sketch_factory.
      cardinality_sketch_factory: A cardinality sketch factory.
    """
    SketchBase.__init__(self)
    # A dictionary that contains multiple sketches, which include:
    # (1) sketches with frequency equal to k, where k < max_freq;
    # (2) a sketch with frequency greater than or equal to max_freq;
    self.sketches = {}
    self.seed = random_seed
    self.max_freq = max_freq
    self.cardinality_sketch_factory = cardinality_sketch_factory

  @classmethod
  def init_from_exact_multi_set(cls, max_freq, exact_multi_set,
                                cardinality_sketch_factory, random_seed):
    """Initialize a Stratified sketch from one ExactMultiSet.

    Args:
      exact_multi_set: ExactMultiSet object to use for initialization.
      CardinalitySketch: Class type of cardinality sketches this stratified
        sketch will hold.
    """
    assert (cardinality_sketch_factory is
            not None), ('cardinality_sketch is None')
    stratified_sketch = cls(
        max_freq=max_freq,
        cardinality_sketch_factory=cardinality_sketch_factory,
        random_seed=random_seed)

    reversedict = {}
    for k, v in exact_multi_set.ids().items():
      reversedict.setdefault(min(v, max_freq), []).append(k)

    for freq in range(1, max_freq):
      stratified_sketch.sketches[freq] = cardinality_sketch_factory(random_seed)
      if (freq in reversedict):
        stratified_sketch.sketches[freq].add_ids(reversedict[freq])

    # Initialize max_freq
    max_key = str(max_freq) + '+'
    stratified_sketch.sketches[max_key] = cardinality_sketch_factory(
        random_seed)
    if (max_freq in reversedict):
      stratified_sketch.sketches[max_key].add_ids(reversedict[max_freq])
    return stratified_sketch

  @classmethod
  def init_from_set_generator(cls, max_freq, set_generator,
                              cardinality_sketch_factory, random_seed):
    """Initialize a Stratified sketch from a Set Generator.

    Args:
      set_generator: SetGenerator object to draw ids from for initialization.
      CardinalitySketch: Class type of cardinality sketches this stratified
        sketch will hold.
    """
    assert (cardinality_sketch_factory is
            not None), ('cardinality_sketch is None')
    exact_multi_set = ExactMultiSet()
    for generated_set in set_generator:
      exact_multi_set.add_ids(generated_set)
    return cls.init_from_exact_multi_set(max_freq, exact_multi_set,
                                         cardinality_sketch_factory,
                                         random_seed)

  def assert_compatible(self, other):
    """"Check if the two StratifiedSketch are comparable.

    Args:
      other: the other StratifiedSketch for comparison.

    Raises:
      AssertionError: if the other sketches are not StratifiedSketch, or if
      their random_seed are different, or if the frequency targets are
      different.
    """
    assert isinstance(other,
                      StratifiedSketch), ('other is not a StratifiedSketch.')
    assert self.seed == other.seed, ('The random seeds are not the same: '
                                     f'{self.seed} != {other.seed}')
    assert self.max_freq == other.max_freq, (
        'The frequency targets are different: '
        f'{self.max_freq} != {other.max_freq}')


class PairwiseEstimator(EstimatorBase):
  """Merge and estimate two StratifiedSketch."""

  def __init__(self, sketch_operator, cardinality_estimator):
    """Create an estimator for two Stratified sketches.

    Args:
      sketch_operator: an object that have union, intersection, and difference
        methods for two sketches.
      cardinality_estimator: a cardinality estimator for estimating the
        cardinality of a sketch.
    """
    self.cardinality_estimator = cardinality_estimator
    self.sketch_union = sketch_operator.union
    self.sketch_difference = sketch_operator.difference
    self.sketch_intersection = sketch_operator.intersection

  def __call__(self, this, that):
    merged = self.merge_sketches(this, that)
    return self.estimate_cardinality(merged)

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
    max_key = str(max_freq) + '+'
    merged_sketch = copy.deepcopy(this)

    this_one_plus = this.cardinality_sketch_factory(0)
    that_one_plus = that.cardinality_sketch_factory(0)
    for k in range(1, max_freq):
      this_one_plus = self.sketch_union(this_one_plus, this.sketches[k])
      that_one_plus = self.sketch_union(that_one_plus, that.sketches[k])

    this_one_plus = self.sketch_union(this_one_plus, this.sketches[max_key])
    that_one_plus = self.sketch_union(that_one_plus, that.sketches[max_key])

    for k in range(1, max_freq):
      # Calculate A(k) & B(0) = A(k) - (A(k) & B(1+))
      merged = self.sketch_difference(
          this.sketches[k],
          self.sketch_intersection(this.sketches[k], that_one_plus))

      # Calculate A(0) & B(k) = B(k) - (B(k) & A(1+))
      merged = self.sketch_union(
          merged,
          self.sketch_difference(
              that.sketches[k],
              self.sketch_intersection(this_one_plus, that.sketches[k])))

      # Calculate A(i) & B(k-i)
      for i in range(1, k):
        merged = self.sketch_union(
            merged,
            self.sketch_intersection(this.sketches[i], that.sketches[(k - i)]))
      merged_sketch.sketches[k] = merged

    # Calculate Merged(max_freq)
    merged = this.sketches[max_key]
    rest = that_one_plus
    for k in range(1, max_freq):
      merged = self.sketch_union(
          merged, self.sketch_intersection(this.sketches[max_freq - k], rest))
      rest = self.sketch_difference(rest, that.sketches[k])

    merged = self.sketch_union(
        merged,
        self.sketch_difference(
            that.sketches[max_key],
            self.sketch_intersection(that.sketches[max_key], this_one_plus)))
    merged_sketch.sketches[max_key] = merged

    # Calculate Merged(1+)
    merged_one_plus = None
    for k in range(1, max_freq):
      merged_one_plus = self.sketch_union(merged_one_plus,
                                          merged_sketch.sketches[k])
      merged_one_plus = self.sketch_union(merged_one_plus,
                                    merged_sketch.sketches[max_key])
    merged_sketch.sketches[ONE_PLUS] = merged_one_plus
    return merged_sketch

  def estimate_cardinality(self, stratified_sketch):
    """Estimate the cardinality of a StratifiedSketch.

    Args:
     stratified_sketch: a StratifiedSketch object.

    Returns:
      A dictionary: the key is the frequency and the value is the corresponding
      cardinality.
    """
    result = {}
    for freq, sketch in stratified_sketch.sketches.items():
      result[freq] = self.cardinality_estimator([sketch])
    return result


class SequentialEstimator(EstimatorBase):
  """Sequential frequency estimator."""

  def __init__(self, sketch_operator, cardinality_estimator):
    self.pairwise_estimator = PairwiseEstimator(sketch_operator,
                                                cardinality_estimator)

  def __call__(self, sketches_list):
    merged = self.merge_sketches(sketches_list)
    return self.pairwise_estimator.estimate_cardinality(merged)

  def merge_sketches(self, sketches_list):
    return functools.reduce(self.pairwise_estimator.merge_sketches,
                            sketches_list)


class ExactSetOperation(object):
  """Set operations for ExactSet."""

  @classmethod
  def union(cls, this, that):
    """Union operation for ExactSet."""
    if this is None:
      return copy.deepcopy(that)
    if that is None:
      return copy.deepcopy(this)
    result = copy.deepcopy(this)
    result_key_set = set(result.ids().keys())
    that_key_set = set(that.ids().keys())
    result._ids = {x: 1 for x in result_key_set.union(that_key_set)}
    return result

  @classmethod
  def intersection(cls, this, that):
    """Intersection operation for ExactSet."""
    if this is None or that is None:
      return None
    result = copy.deepcopy(this)
    result_key_set = set(result.ids().keys())
    that_key_set = set(that.ids().keys())
    result._ids = {x: 1 for x in result_key_set.intersection(that_key_set)}
    return result

  @classmethod
  def difference(cls, this, that):
    """Difference operation for ExactSet."""
    if this is None:
      return None
    if that is None:
      return copy.deepcopy(this)
    result = copy.deepcopy(this)
    result_key_set = set(result.ids().keys())
    that_key_set = set(that.ids().keys())
    result._ids = {x: 1 for x in result_key_set.difference(that_key_set)}
    return result
