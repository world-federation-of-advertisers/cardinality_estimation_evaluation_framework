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
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator

ONE_PLUS = '1+'

class StratifiedTest(parameterized.TestCase):

  def test_sketch_building_from_exact_multi_set(self):
    max_freq = 3
    expected = ExactMultiSet()
    for k in range(max_freq + 2):
        for i in range(k):
            expected.add(k)

    s = stratified_sketch.StratifiedSketch(sketch_list=[], max_freq=max_freq,
          random_seed=np.random.RandomState(1))
    s.init_from_exact_multi_set(expected, CardinalitySketch=ExactMultiSet)

    self.assertLen(s.sketches.keys(), max_freq+1)
    self.assertLen(s.sketches[ONE_PLUS], len(expected))

    for id, freq in expected.ids().items():
      freq_key = str(min(freq, max_freq))
      self.assertTrue(id in s.sketches[freq_key])

  def test_sketch_building_from_set_generator(self):
    universe_size = 1000
    set_sizes = [100] * 5
    max_freq = 3

    ind_gen = set_generator.IndependentSetGenerator(
      universe_size=universe_size, set_sizes=set_sizes,
      random_state=np.random.RandomState(1))
    ind_gen_clone = set_generator.IndependentSetGenerator(
      universe_size=universe_size, set_sizes=set_sizes,
      random_state=np.random.RandomState(1))
    s = stratified_sketch.StratifiedSketch(sketch_list=[], max_freq=max_freq,
          random_seed=np.random.RandomState(1))


    expected = ExactMultiSet()
    for set_clone in ind_gen_clone:
        for clone_id in set_clone:
            expected.add(clone_id)

    s.init_from_set_generator(set_generator=ind_gen, CardinalitySketch=ExactMultiSet)
    self.assertLen(s.sketches.keys(), max_freq+1)
    self.assertLen(s.sketches[ONE_PLUS], len(ind_gen_clone.union_ids))
    for id, freq in expected.ids().items():
        freq_key = min(freq, max_freq)
        self.assertTrue(id in s.sketches[str(freq_key)])

  @parameterized.parameters(
      ((1, 2), (1, 1)),
      ((1, 1), (1, 2)))
  def test_assert_compatible(self, max_freq, random_seed):
    stratified_sketch_list = []
    for i in range(2):
      s = stratified_sketch.StratifiedSketch(
          sketch_list=[],
          max_freq=max_freq[i],
          random_seed=random_seed[i])
      stratified_sketch_list.append(s)
    with self.assertRaises(AssertionError):
      stratified_sketch_list[0].assert_compatible(stratified_sketch_list[1])

class PairwiseEstimatorTest(absltest.TestCase):

  def generate_multi_set(self, tuple_list):
    multi_set = ExactMultiSet()
    for tuple in tuple_list:
      for id in range(tuple[1]):
        multi_set.add(tuple[0])
    return multi_set

  def test_merge_sketches(self):
    max_freq = 3
    this_multi_set = self.generate_multi_set([(1,2),(2,3),(3,1)])
    that_multi_set = self.generate_multi_set([(1,1),(3,1),(4,5),(5,1)])
    expected = {
    ONE_PLUS : {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
    '1' : {5: 1},
    '2' : {3: 1},
    '3' : {1: 1, 2: 1, 4: 1},
    }

    this_sketch = stratified_sketch.StratifiedSketch(
      sketch_list=[], max_freq=max_freq)
    that_sketch = stratified_sketch.StratifiedSketch(
      sketch_list=[], max_freq=max_freq)
    this_sketch.init_from_exact_multi_set(this_multi_set,
      CardinalitySketch=ExactMultiSet)
    that_sketch.init_from_exact_multi_set(that_multi_set,
      CardinalitySketch=ExactMultiSet)
    estimator = stratified_sketch.PairwiseEstimator(
      sketch_operator=stratified_sketch.ExactMultiSetOperation)

    merged_sketches = estimator.merge_sketches(
        this_sketch, that_sketch)

    self.assertLen(merged_sketches.sketches, len(expected))
    self.assertEqual(merged_sketches.sketches[ONE_PLUS].ids(), expected[ONE_PLUS])
    for freq_key, sketch in expected.items():
      self.assertEqual(merged_sketches.sketches[freq_key].ids(), sketch)

class SequentialEstimatorTest(absltest.TestCase):
  def generate_multi_set(self, tuple_list):
    multi_set = ExactMultiSet()
    for tuple in tuple_list:
      for id in range(tuple[1]):
        multi_set.add(tuple[0])
    return multi_set

  def generate_sketches_from_sets(self, multi_sets, max_freq):
    sketches = []
    for multi_set in multi_sets:
      s = stratified_sketch.StratifiedSketch(
        sketch_list=[], max_freq=max_freq)
      sketches.append(s)
      s.init_from_exact_multi_set(multi_set, CardinalitySketch=ExactMultiSet)
    return sketches

  def test_merge_sketches(self):
    max_freq = 3
    init_set_list = []
    init_set_list.append(self.generate_multi_set([(1,2),(2,3),(3,1)]))
    init_set_list.append(self.generate_multi_set([(1,1),(3,1),(4,5),(5,1)]))
    init_set_list.append(self.generate_multi_set([(5,1),(6,1)]))
    sketches_to_merge = self.generate_sketches_from_sets(init_set_list, max_freq)

    estimator = stratified_sketch.SequentialEstimator(
        sketch_operator=stratified_sketch.ExactMultiSetOperation)
    merged_sketches = estimator.merge_sketches(sketches_to_merge)

    expected = {
      ONE_PLUS : {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
      '1' : {6: 1},
      '2' : {3: 1, 5: 1},
      '3' : {1: 1, 2: 1, 4: 1},
    }

    self.assertLen(merged_sketches.sketches, len(expected))
    for freq, sketch in expected.items():
      self.assertEqual(merged_sketches.sketches[freq].ids(), sketch)

if __name__ == '__main__':
  absltest.main()
