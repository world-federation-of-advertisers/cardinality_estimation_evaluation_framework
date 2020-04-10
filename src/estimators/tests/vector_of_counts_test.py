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

"""Tests for wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import IdentityNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import LaplaceNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import PairwiseEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import SequentialEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts


class VectorOfCountsTest(absltest.TestCase):

  def test_add_ids(self):
    sketch = VectorOfCounts(num_buckets=8, random_seed=0)
    sketch.add_ids([1])
    self.assertEqual(sketch.cardinality(), 1)

  def test_add_ids_multiple_times(self):
    sketch = VectorOfCounts(num_buckets=8, random_seed=0)
    sketch.add_ids([1])
    with self.assertRaises(AssertionError):
      sketch.add_ids([1])

  def test_add_ids_random_state(self):
    sketch1 = VectorOfCounts(num_buckets=8, random_seed=0)
    sketch1.add_ids(range(4))
    sketch2 = VectorOfCounts(num_buckets=8, random_seed=0)
    sketch2.add_ids(range(4))
    np.testing.assert_array_equal(
        sketch1.stats,
        sketch2.stats,
        err_msg='Two VoC are not the same with the same random seed.')


class NoiserTest(absltest.TestCase):

  def test_identity(self):
    noiser = IdentityNoiser()
    sketch_original = VectorOfCounts(num_buckets=8, random_seed=1)
    sketch_noised = noiser(sketch_original)
    np.testing.assert_array_equal(
        sketch_original.stats,
        sketch_noised.stats,
        err_msg='IdentityNoiser should not change the sketch.')

  def test_laplace(self):
    noiser = LaplaceNoiser()
    sketch_original = VectorOfCounts(num_buckets=8, random_seed=1)
    sketch_noised = noiser(sketch_original)
    for o, n in zip(sketch_original.stats, sketch_noised.stats):
      self.assertNotEqual(o, n)


class PairwiseEstimatorTest(absltest.TestCase):

  def test_assert_compatible_not_vector_of_count(self):
    sketch = VectorOfCounts(num_buckets=4, random_seed=2)
    estimator = PairwiseEstimator()
    with self.assertRaises(AssertionError):
      estimator.assert_compatible(sketch, [])
    with self.assertRaises(AssertionError):
      estimator.assert_compatible([], sketch)

  def test_assert_compatible_not_equal_length(self):
    sketch1 = VectorOfCounts(num_buckets=4, random_seed=2)
    sketch2 = VectorOfCounts(num_buckets=8, random_seed=2)
    estimator = PairwiseEstimator()
    with self.assertRaises(AssertionError):
      estimator.assert_compatible(sketch1, sketch2)

  def test_assert_compatible_not_same_hash_function(self):
    sketch1 = VectorOfCounts(num_buckets=4, random_seed=1)
    sketch2 = VectorOfCounts(num_buckets=4, random_seed=2)
    estimator = PairwiseEstimator()
    with self.assertRaises(AssertionError):
      estimator.assert_compatible(sketch1, sketch2)

  def test_merge(self):
    sketch_list = []
    for _ in range(2):
      sketch = VectorOfCounts(num_buckets=2, random_seed=2)
      sketch.add_ids([1])
      sketch_list.append(sketch)
    merged = PairwiseEstimator.merge(sketch_list[0], sketch_list[1])
    np.testing.assert_array_equal(
        np.sort(merged.stats),
        np.array([0, 1.5]))


class SequentialEstimatorTest(absltest.TestCase):

  def test_estimate_cardinality(self):
    sketch_list = []
    for _ in range(3):
      sketch = VectorOfCounts(num_buckets=2, random_seed=3)
      sketch.add_ids([1])
      sketch_list.append(sketch)
    estimator = SequentialEstimator()
    result = estimator(sketch_list)
    actual = 1.75
    self.assertEqual(result, actual)


if __name__ == '__main__':
  absltest.main()
