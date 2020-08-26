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
"""Tests for wfa_cardinality_estimation_evaluation_framework.estimators.stratified_sketch."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import farmhash
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators import stratified_sketch
from wfa_cardinality_estimation_evaluation_framework.estimators.stratified_sketch import ONE_PLUS
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


def generate_multi_set(tuple_list):
  multi_set = ExactMultiSet()
  for tuple in tuple_list:
    for id in range(tuple[1]):
      multi_set.add(tuple[0])
  return multi_set


class PlusNoiser(object):

  def __init__(self, epsilon, random_state=None):
    self.constant = epsilon * 10

  def __call__(self, sketch):
    noised_sketch = ExactMultiSet()
    for x in sketch.ids():
      noised_sketch.add(x + self.constant)
    return noised_sketch


class MinusDenoiser(object):

  def __init__(self, epsilon, random_state=None):
    self.constant = epsilon * 10

  def __call__(self, sketch):
    denoised_sketch = ExactMultiSet()
    for x in sketch.ids():
      denoised_sketch.add(x - 2 * self.constant)
    return denoised_sketch


class FakeSetGenerator(set_generator.SetGeneratorBase):
  """Generator for a fixed collection of sets."""

  def __init__(self, set_list):
    self.set_list = set_list

  def __iter__(self):
    for s in self.set_list:
      yield s
    return self


class StratifiedTest(parameterized.TestCase):
  """Test StratifiedSketch creationg."""

  def test_sketch_building_from_exact_multi_set(self):
    max_freq = 3
    input_set = ExactMultiSet()
    for k in range(max_freq + 2):
      for i in range(k):
        input_set.add(k)

    expected = {1: {1: 1}, 2: {2: 1}, '3+': {3: 1, 4: 1}}

    s = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        input_set,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)

    self.assertLen(s.sketches.keys(), max_freq)

    for freq, sketch in s.sketches.items():
      self.assertEqual(sketch.ids(), expected[freq])

  def test_sketch_create_and_destroy(self):
    max_freq = 3
    s = stratified_sketch.StratifiedSketch(
        max_freq=max_freq,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)

    for k in range(max_freq + 2):
      for i in range(k):
        s.add(k)

    s.create_sketches()

    expected = {1: {1: 1}, 2: {2: 1}, '3+': {3: 1, 4: 1}}

    self.assertLen(s.sketches.keys(), max_freq)

    for freq, sketch in s.sketches.items():
      self.assertEqual(sketch.ids(), expected[freq])

    s._destroy_sketches()
    self.assertEqual(s.sketches, {})

  def test_noise_with_oneplus_budget(self):
    max_freq = 3
    s = stratified_sketch.StratifiedSketch(
        max_freq=max_freq,
        epsilon=0.8,
        epsilon_split=0.5,
        noiser_class=PlusNoiser,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)

    vids = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    for vid in vids:
      s.add(vid)

    s.create_sketches()
    s.create_one_plus_sketch(stratified_sketch.ExactSetOperator.union)
    s.noise()

    expected = {
        '1+': {
            5.0: 1,
            6.0: 1,
            7.0: 1,
            8.0: 1
        },
        1: {
            5.0: 1
        },
        2: {
            6.0: 1
        },
        '3+': {
            7.0: 1,
            8.0: 1
        }
    }

    self.assertLen(s.sketches.keys(), len(expected.keys()))

    for freq, sketch in s.sketches.items():
      self.assertEqual(sketch.ids(), expected[freq])

    s._destroy_sketches()
    self.assertEqual(s.sketches, {})

  def test_noise_without_oneplus_budget(self):
    max_freq = 3
    s = stratified_sketch.StratifiedSketch(
        max_freq=max_freq,
        epsilon=0.8,
        epsilon_split=0,
        noiser_class=PlusNoiser,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)

    vids = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    for vid in vids:
      s.add(vid)

    s.create_sketches()
    s.create_one_plus_sketch(stratified_sketch.ExactSetOperator.union)
    s.noise()

    expected = {
        '1+': {
            9.0: 1,
            10.0: 1,
            11.0: 1,
            12.0: 1
        },
        1: {
            9.0: 1
        },
        2: {
            10.0: 1
        },
        '3+': {
            11.0: 1,
            12.0: 1
        }
    }

    self.assertLen(s.sketches.keys(), len(expected.keys()))

    for freq, sketch in s.sketches.items():
      self.assertEqual(sketch.ids(), expected[freq])

    s._destroy_sketches()
    self.assertEqual(s.sketches, {})

  def test_sketch_building_from_set_generator(self):
    universe_size = 1000
    set_sizes = [100] * 5
    max_freq = 3

    expected_sets = [[1, 1, 1, 2, 2, 3], [1, 1, 1, 3, 3, 4]]
    set_gen = FakeSetGenerator(expected_sets)

    s = stratified_sketch.StratifiedSketch.init_from_set_generator(
        max_freq,
        set_generator=set_gen,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)

    expected = {1: {4: 1}, 2: {2: 1}, '3+': {1: 1, 3: 1}}
    self.assertLen(s.sketches.keys(), max_freq)

    for freq, sketch in s.sketches.items():
      self.assertEqual(sketch.ids(), expected[freq])

  @parameterized.parameters(((1, 2), (1, 1)), ((1, 1), (1, 2)))
  def test_assert_compatible(self, max_freq, random_seed):
    stratified_sketch_list = []
    for i in range(2):
      s = stratified_sketch.StratifiedSketch(
          cardinality_sketch_factory=None,
          underlying_set=None,
          max_freq=max_freq[i],
          random_seed=random_seed[i])
      stratified_sketch_list.append(s)
    with self.assertRaises(AssertionError):
      stratified_sketch_list[0].assert_compatible(stratified_sketch_list[1])


class PairwiseEstimatorTest(absltest.TestCase):
  """Test PairwiseEstimator merge and frequency estimation."""

  def setUp(self):
    super(PairwiseEstimatorTest, self).setUp()
    max_freq = 3
    this_multi_set = generate_multi_set([(1, 2), (2, 3), (3, 1), (10, 1)])
    that_multi_set = generate_multi_set([(1, 1), (3, 1), (4, 5), (5, 1)])
    self.this_sketch = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        this_multi_set,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)
    self.that_sketch = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        that_multi_set,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)
    self.estimator = stratified_sketch.PairwiseEstimator(
        sketch_operator=stratified_sketch.ExactSetOperator,
        cardinality_estimator=LosslessEstimator())
    self.merge_expected = {
        ONE_PLUS: {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            10: 1
        },
        1: {
            5: 1,
            10: 1
        },
        2: {
            3: 1
        },
        '3+': {
            1: 1,
            2: 1,
            4: 1
        },
    }

  def test_merge_sketches(self):

    merged_sketches = self.estimator.merge(self.this_sketch, self.that_sketch)

    self.assertLen(merged_sketches.sketches, len(self.merge_expected))
    self.assertEqual(merged_sketches.sketches[ONE_PLUS].ids(),
                     self.merge_expected[ONE_PLUS])
    for freq_key, sketch in self.merge_expected.items():
      self.assertEqual(merged_sketches.sketches[freq_key].ids(), sketch)

  def test_merge_destroyed_sketches(self):

    self.this_sketch._destroy_sketches()
    self.that_sketch._destroy_sketches()
    merged_sketches = self.estimator.merge(self.this_sketch, self.that_sketch)
    self.assertLen(merged_sketches.sketches, len(self.merge_expected))
    self.assertEqual(merged_sketches.sketches[ONE_PLUS].ids(),
                     self.merge_expected[ONE_PLUS])
    for freq_key, sketch in self.merge_expected.items():
      self.assertEqual(merged_sketches.sketches[freq_key].ids(), sketch)

  def test_end_to_end(self):
    estimated = self.estimator(self.this_sketch, self.that_sketch)
    expected = [6, 4, 3]
    self.assertEqual(estimated, expected)

  def test_end_to_end_noise_with_oneplus_budget(self):
    max_freq = 3
    this_multi_set = generate_multi_set([(1, 2), (2, 3), (3, 1), (10, 1)])
    that_multi_set = generate_multi_set([(1, 1), (3, 1), (4, 5), (5, 1)])
    this_noised_sketch = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        this_multi_set,
        epsilon=0.8,
        epsilon_split=0.5,
        noiser_class=PlusNoiser,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)
    that_noised_sketch = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        that_multi_set,
        epsilon=0.8,
        epsilon_split=0.5,
        noiser_class=PlusNoiser,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)
    estimator = stratified_sketch.PairwiseEstimator(
        denoiser_class=MinusDenoiser,
        sketch_operator=stratified_sketch.ExactSetOperator,
        cardinality_estimator=LosslessEstimator())

    estimated = estimator(this_noised_sketch, that_noised_sketch)
    expected = [6, 4, 3]
    self.assertEqual(estimated, expected)


  def test_end_to_end_noise_without_oneplus_budget(self):
    max_freq = 3
    this_multi_set = generate_multi_set([(1, 2), (2, 3), (3, 1), (10, 1)])
    that_multi_set = generate_multi_set([(1, 1), (3, 1), (4, 5), (5, 1)])
    this_noised_sketch = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        this_multi_set,
        epsilon=0.8,
        epsilon_split=0,
        noiser_class=PlusNoiser,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)
    that_noised_sketch = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
        max_freq,
        that_multi_set,
        epsilon=0.8,
        epsilon_split=0,
        noiser_class=PlusNoiser,
        cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
        random_seed=1)
    estimator = stratified_sketch.PairwiseEstimator(
        denoiser_class=MinusDenoiser,
        sketch_operator=stratified_sketch.ExactSetOperator,
        cardinality_estimator=LosslessEstimator())

    estimated = estimator(this_noised_sketch, that_noised_sketch)
    expected = [6, 4, 3]
    self.assertEqual(estimated, expected)

class SequentialEstimatorTest(absltest.TestCase):
  """Test SequentialEstimator merge and frequency estimation."""

  def generate_sketches_from_sets(self, multi_sets, max_freq):
    sketches = []
    for multi_set in multi_sets:
      s = stratified_sketch.StratifiedSketch.init_from_exact_multi_set(
          max_freq,
          multi_set,
          cardinality_sketch_factory=ExactMultiSet.get_sketch_factory(),
          random_seed=1)
      sketches.append(s)
    return sketches

  def setUp(self):
    super(SequentialEstimatorTest, self).setUp()
    max_freq = 3
    init_set_list = []
    init_set_list.append(generate_multi_set([(1, 2), (2, 3), (3, 1)]))
    init_set_list.append(generate_multi_set([(1, 1), (3, 1), (4, 5), (5, 1)]))
    init_set_list.append(generate_multi_set([(5, 1)]))
    self.sketches_list = self.generate_sketches_from_sets(
        init_set_list, max_freq)

  def test_merge_sketches(self):
    estimator = stratified_sketch.SequentialEstimator(
        sketch_operator=stratified_sketch.ExactSetOperator,
        cardinality_estimator=LosslessEstimator())
    merged_sketches = estimator.merge_sketches(self.sketches_list)

    expected = {
        ONE_PLUS: {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1
        },
        1: {},
        2: {
            3: 1,
            5: 1
        },
        '3+': {
            1: 1,
            2: 1,
            4: 1
        },
    }

    self.assertLen(merged_sketches.sketches, len(expected))
    for freq, sketch in expected.items():
      self.assertEqual(merged_sketches.sketches[freq].ids(), sketch)

  def test_end_to_end(self):
    estimator = stratified_sketch.SequentialEstimator(
        sketch_operator=stratified_sketch.ExactSetOperator,
        cardinality_estimator=LosslessEstimator())
    estimated = estimator(self.sketches_list)

    expected = [5, 5, 3]
    self.assertEqual(estimated, expected)


class ExactSetOperatorTest(absltest.TestCase):
  """Test ExactSet set operations."""

  def setUp(self):
    super().setUp()
    self.this_set = generate_multi_set([(1, 1), (2, 1), (3, 1)])
    self.that_set = generate_multi_set([(1, 1), (2, 1), (4, 1), (5, 1)])

  def test_union(self):
    expected_union_set = generate_multi_set([(1, 1), (2, 1), (3, 1), (4, 1),
                                             (5, 1)])
    union_set = stratified_sketch.ExactSetOperator.union(
        self.this_set, self.that_set)
    self.assertEqual(union_set.ids(), expected_union_set.ids())

  def test_intersection(self):
    expected_intersection_set = generate_multi_set([(1, 1), (2, 1)])
    intersection_set = stratified_sketch.ExactSetOperator.intersection(
        self.this_set, self.that_set)
    self.assertEqual(intersection_set.ids(), expected_intersection_set.ids())

  def test_difference(self):
    expected_difference_set = generate_multi_set([(3, 1)])
    difference_set = stratified_sketch.ExactSetOperator.difference(
        self.this_set, self.that_set)
    self.assertEqual(difference_set.ids(), expected_difference_set.ids())


if __name__ == '__main__':
  absltest.main()
