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

from absl import logging
import copy
import functools
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet

ONE_PLUS = '1+'


class ExactSetOperator(object):
  """Set operations for ExactSet.

  The methods below all accept an ExactMultiSet object and returning an
  ExactMultiSet object.
  """

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


class StratifiedSketch(SketchBase):
  """A frequency sketch that contains cardinality sketches per frequency bucket."""

  @classmethod
  def get_sketch_factory(cls,
                         max_freq,
                         cardinality_sketch_factory,
                         noiser_class=None,
                         epsilon=0,
                         epsilon_split=0.5,
                         underlying_set=None,
                         union=ExactSetOperator.union):

    def f(random_seed):
      return cls(
          max_freq=max_freq,
          cardinality_sketch_factory=cardinality_sketch_factory,
          random_seed=random_seed,
          underlying_set=underlying_set,
          noiser_class=noiser_class,
          epsilon=epsilon,
          epsilon_split=epsilon_split,
          union=union)

    return f

  def __init__(self,
               max_freq,
               cardinality_sketch_factory,
               random_seed,
               noiser_class=None,
               epsilon=0,
               epsilon_split=0.5,
               underlying_set=None,
               union=ExactSetOperator.union):
    """Construct a Stratified sketch.

    Args:
      max_freq: the maximum targeting frequency level. For example, if it is set
        to 3, then the sketches will include frequency=1, 2, 3+ (frequency >=
        3).
      cardinality_sketch_factory: A cardinality sketch factory.
      random_seed: This arg exists in order to conform to
        simulator.EstimatorConfig.sketch_factory.
      noiser : A noiser class that is a subclass of base.EstimateNoiserBase.
      epsilon : Total privacy budget to spend for noising this sketch.
      epsilon_split : Ratio of privacy budget to spend to noise 1+ sketch.
      underlying_set : ExactMultiSet object that holds the frequency for each
        item for this Stratified Sketch.
      union : Function to be used to calculate the 1+ sketch as the union of the
        others.
    """
    SketchBase.__init__(self)
    # A dictionary that contains multiple sketches, which include:
    # (1) sketches with frequency equal to k, where k < max_freq;
    # (2) a sketch with frequency greater than or equal to max_freq;
    assert (epsilon_split >= 0 and
            epsilon_split < 1), ('epsilon split is not between 0 and 1')

    self.sketches = {}
    self.seed = random_seed
    self.max_freq = max_freq
    self.cardinality_sketch_factory = cardinality_sketch_factory
    self.underlying_set = underlying_set if underlying_set is not None else ExactMultiSet(
    )
    self.epsilon_split = epsilon_split
    self.epsilon = epsilon
    self.union = union
    self.one_plus_noiser = None
    self.rest_noiser = None

    if noiser_class is not None:
      if epsilon_split != 0:
        self.one_plus_noiser = noiser_class(epsilon=epsilon * epsilon_split)
        self.rest_noiser = noiser_class(epsilon=epsilon * (1 - epsilon_split))
      else:
        self.one_plus_noiser = noiser_class(epsilon=epsilon)
        self.rest_noiser = noiser_class(epsilon=epsilon)

  def create_frequency_sketches(self):
    if self.sketches != {}:
      return

    reversedict = {}
    for k, v in self.underlying_set.ids().items():
      reversedict.setdefault(min(v, self.max_freq), []).append(k)

    for freq in range(1, self.max_freq):
      self.sketches[freq] = self.cardinality_sketch_factory(self.seed)
      if (freq in reversedict):
        self.sketches[freq].add_ids(reversedict[freq])

    # Initialize max_freq
    max_key = str(self.max_freq) + '+'
    self.sketches[max_key] = self.cardinality_sketch_factory(self.seed)
    if (self.max_freq in reversedict):
      self.sketches[max_key].add_ids(reversedict[self.max_freq])

  def create_one_plus_with_merge(self):
    assert (self.union is not None), (
        '1+ cannot be calculated because union operation is not known')
    one_plus = self.cardinality_sketch_factory(self.seed)
    max_key = str(self.max_freq) + '+'
    for k in range(1, self.max_freq):
      one_plus = self.union(one_plus, self.sketches[k])
    one_plus = self.union(one_plus, self.sketches[max_key])
    return one_plus

  def create_one_plus_from_underlying(self):
    one_plus = self.cardinality_sketch_factory(self.seed)
    for k, v in self.underlying_set.ids().items():
      one_plus.add_ids([k])
    return one_plus

  def create_one_plus_sketch(self):
    """Create the 1+ sketch for this stratified sketch.

       We support creation of 1+ sketch for 2 scenarios :
         1) 1+ sketch is created from the underlying exact set directly. Here we
         noise 1+ sketch with epsilon = (self.epsilon * self.epsilon_split).

         2) 1+ sketch is created from the union of all other frequencies. Here
         we noise 1+ sketch with epsilon = self.epsilon

      These two scenarios are controlled with the epsilon_split parameter. If
      epsilon_split = 0, then do scenario 1 otherwise do scenario 2.
    """

    if ONE_PLUS in self.sketches:
      return

    assert (self.epsilon_split >= 0 and self.epsilon_split < 1), (
        'epsilon split is not between 0 and 1 for ONE_PLUS sketch creation')

    if (self.epsilon_split == 0):
      self.sketches[ONE_PLUS] = self.create_one_plus_with_merge()
    else:
      self.sketches[ONE_PLUS] = self.create_one_plus_from_underlying()

  def create_sketches(self):
    self.create_frequency_sketches()
    self.create_one_plus_sketch()
    self.noise()

  def noise(self):
    max_key = str(self.max_freq) + '+'
    if self.rest_noiser is not None:
      for i in range(1, self.max_freq):
        self.sketches[i] = self.rest_noiser(self.sketches[i])
      self.sketches[max_key] = self.rest_noiser(self.sketches[max_key])

    if self.one_plus_noiser is not None:
      self.sketches[ONE_PLUS] = self.one_plus_noiser(self.sketches[ONE_PLUS])

  def _destroy_sketches(self):
    self.sketches = {}

  def add(self, x):
    if (self.sketches != {}):
      self._destroy_sketches()
      logging.warn(
          """Tried to add ids after sketch creation, sketches are destroyed.""",
          RuntimeWarning)
    self.underlying_set.add(x)

  @classmethod
  def init_from_exact_multi_set(cls,
                                max_freq,
                                exact_multi_set,
                                cardinality_sketch_factory,
                                random_seed,
                                noiser_class=None,
                                epsilon=0,
                                epsilon_split=0.5,
                                union=ExactSetOperator.union):
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
        underlying_set=exact_multi_set,
        cardinality_sketch_factory=cardinality_sketch_factory,
        random_seed=random_seed,
        noiser_class=noiser_class,
        epsilon=epsilon,
        epsilon_split=epsilon_split,
        union=union)
    stratified_sketch.create_sketches()

    return stratified_sketch

  @classmethod
  def init_from_set_generator(cls,
                              max_freq,
                              set_generator,
                              cardinality_sketch_factory,
                              random_seed,
                              noiser_class=None,
                              epsilon=0,
                              epsilon_split=0.5,
                              union=ExactSetOperator.union):
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
    return cls.init_from_exact_multi_set(
        max_freq,
        exact_multi_set,
        cardinality_sketch_factory,
        random_seed,
        noiser_class=noiser_class,
        epsilon=epsilon,
        epsilon_split=epsilon_split,
        union=union)

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
    assert isinstance(self.cardinality_sketch_factory,
                      type(other.cardinality_sketch_factory))
    if (self.sketches != {} and other.sketches != {}):
      assert isinstance(
          list(self.sketches.values())[0],
          type(list(other.sketches.values())[0]))


class PairwiseEstimator(EstimatorBase):
  """Merge and estimate two StratifiedSketch."""

  def __init__(self,
               sketch_operator,
               cardinality_estimator,
               denoiser_class=None):
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
    self.denoiser_class = denoiser_class

  def __call__(self, this, that):
    merged = self.merge(this, that)
    return self.estimate_cardinality(merged)

  def merge(self, this, that):
    denoised_this = self.prepare_sketch(this)
    denoised_that = self.prepare_sketch(that)
    merged = self.merge_sketches(denoised_this, denoised_that)
    return merged

  def prepare_sketch(self, this):
    assert isinstance(this, StratifiedSketch)

    this.create_sketches()

    if self.denoiser_class is not None:
      return self.denoise_sketch(this)

    return copy.deepcopy(this)

  def denoise_sketch(self, stratified_sketch):
    """Denoise a StratifiedSketch.

    Args:
       stratified_sketch: a StratifiedSketch.

    Returns:
      A denoised StratifiedSketch.
    """

    denoised_stratified_sketch = copy.deepcopy(stratified_sketch)

    if self.denoiser_class is None:
      return denoised_stratified_sketch

    epsilon = stratified_sketch.epsilon
    epsilon_split = stratified_sketch.epsilon_split
    max_key = str(stratified_sketch.max_freq) + '+'

    one_plus_epsilon = epsilon if epsilon_split == 0 else epsilon * epsilon_split

    rest_epsilon = stratified_sketch.epsilon * (1 -
                                                stratified_sketch.epsilon_split)

    one_plus_denoiser = self.denoiser_class(epsilon=one_plus_epsilon)
    rest_denoiser = self.denoiser_class(epsilon=rest_epsilon)

    for freq in range(1, denoised_stratified_sketch.max_freq):
      denoised_stratified_sketch.sketches[freq] = rest_denoiser(
          stratified_sketch.sketches[freq])

    denoised_stratified_sketch.sketches[max_key] = rest_denoiser(
        stratified_sketch.sketches[max_key])

    denoised_stratified_sketch.sketches[ONE_PLUS] = one_plus_denoiser(
        stratified_sketch.sketches[ONE_PLUS])

    return denoised_stratified_sketch

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

    this.assert_compatible(that)
    this_one_plus = this.sketches[ONE_PLUS]
    that_one_plus = that.sketches[ONE_PLUS]

    max_freq = this.max_freq
    max_key = str(max_freq) + '+'

    merged_sketch = copy.deepcopy(this)

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

    # We estimate a histogram for each frequency bucket. Since an estimator
    # returns a histogram, we assert that for each bucket it has a lenghth of 1
    # This has to be the case because, the input to the underliying estimators
    # are cardinality sketches and should not have any repeated ids, thus, no
    # bucket other than frequency = 1.
    # We then put them into a list and take the cumilative of it to match the
    # api output.

    result = []
    for freq in range(1, stratified_sketch.max_freq):
      freq_count_histogram = self.cardinality_estimator(
          [stratified_sketch.sketches[freq]])
      assert (len(freq_count_histogram) == 1), (
          'cardinality sketch has more than 1 freq bucket.')
      result.append(freq_count_histogram[0])

    max_key = str(stratified_sketch.max_freq) + '+'
    max_freq_count_histogram = self.cardinality_estimator(
        [stratified_sketch.sketches[max_key]])
    assert (len(max_freq_count_histogram) == 1), (
        'cardinality sketch has more than 1 freq bucket for max_freq.')
    result.append(max_freq_count_histogram[0])
    result = list(np.cumsum(list(reversed(result))))
    result = list(reversed(result))
    return result


class SequentialEstimator(EstimatorBase):
  """Sequential frequency estimator."""

  def __init__(self,
               sketch_operator,
               cardinality_estimator,
               denoiser_class=None):
    self.pairwise_estimator = PairwiseEstimator(
        sketch_operator, cardinality_estimator, denoiser_class=denoiser_class)

  def __call__(self, sketches_list):
    for i, sketch in enumerate(sketches_list):
      sketches_list[i] = self.pairwise_estimator.prepare_sketch(sketch)
    merged = self.merge(sketches_list)
    return self.pairwise_estimator.estimate_cardinality(merged)

  def merge(self, sketches_list):
    return functools.reduce(self.pairwise_estimator.merge_sketches,
                            sketches_list)
