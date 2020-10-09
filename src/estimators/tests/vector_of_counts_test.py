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
from absl.testing import parameterized
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


class PairwiseEstimatorTest(parameterized.TestCase):

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

  def test_has_zero_intersection(self):
    pairwise_estimator = PairwiseEstimator()
    this = VectorOfCounts(num_buckets=64, random_seed=2)
    this.add_ids(range(100))
    # Clip relies on hypothesis testing and hence requires a minimum size
    that = VectorOfCounts(num_buckets=64, random_seed=2)
    that.add_ids(range(100, 200))
    intersection_cardinality = pairwise_estimator._intersection(this, that)
    self.assertTrue(
        pairwise_estimator.has_zero_intersection(
            intersection_cardinality, this, that))

  def test_has_full_intersection(self):
    pairwise_estimator = PairwiseEstimator()
    this = VectorOfCounts(num_buckets=64, random_seed=2)
    this.add_ids(range(100))
    that = VectorOfCounts(num_buckets=64, random_seed=2)
    that.add_ids(range(100))
    intersection_cardinality = pairwise_estimator._intersection(this, that)
    self.assertTrue(
        pairwise_estimator.has_full_intersection(
            intersection_cardinality, this, that))

  def test_merge_no_clip(self):
    sketch_list = []
    for _ in range(2):
      sketch = VectorOfCounts(num_buckets=2, random_seed=2)
      sketch.add_ids([1])
      sketch_list.append(sketch)
    pairwise_estimator = PairwiseEstimator()
    merged = pairwise_estimator.merge(sketch_list[0], sketch_list[1])
    np.testing.assert_array_equal(
        np.sort(merged.stats),
        np.array([0, 1.5]))

  def test_merge_with_clip(self):
    this_sketch = VectorOfCounts(num_buckets=64, random_seed=2)
    this_sketch.add_ids(range(100))
    # First test no intersection
    that_sketch = VectorOfCounts(num_buckets=64, random_seed=2)
    that_sketch.add_ids(range(100, 200))
    pairwise_estimator = PairwiseEstimator(clip=True)
    merged = pairwise_estimator.merge(this_sketch, that_sketch)
    np.testing.assert_array_equal(
        x=merged.stats, y=this_sketch.stats + that_sketch.stats,
        err_msg='Fail to detect the no-intersection case.')
    # Then test full intersection
    that_sketch = VectorOfCounts(num_buckets=64, random_seed=2)
    that_sketch.add_ids(range(100))
    merged = pairwise_estimator.merge(this_sketch, that_sketch)
    np.testing.assert_array_equal(
        x=merged.stats, y=this_sketch.stats,
        err_msg='Fail to detect the full-intersection case.')

  @parameterized.parameters(
      (1, 2),
      (0.5, 4))
  def test_get_std_of_sketch_sum(self, epsilon, expected):
    sketch = VectorOfCounts(num_buckets=2, random_seed=0)
    sketch.stats = np.array([2, 2])
    pairwise_estimator = PairwiseEstimator(clip=True, epsilon=epsilon)
    res = pairwise_estimator._get_std_of_sketch_sum(sketch)
    self.assertEqual(res, expected)

  @parameterized.parameters(
      (1, 3, [0, 0]),
      (1, 1.5, [2, 2]),
      (0.5, 1.5, [0, 0]))
  def test_clip_empty_vector_of_count(self, epsilon, clip_threshold, expected):
    sketch = VectorOfCounts(num_buckets=2, random_seed=0)
    sketch.stats = np.array([2, 2])
    pairwise_estimator = PairwiseEstimator(
        clip=True, epsilon=epsilon, clip_threshold=clip_threshold)
    res = pairwise_estimator.clip_empty_vector_of_count(sketch)
    np.testing.assert_array_equal(res.stats, expected)

  @parameterized.parameters(
      (1, 0, 6),
      (1, 4, 6.325),
      (0.5, 0, 18),
      (0.5, 4, 18.111))
  def test_get_std_of_intersection(
      self, epsilon, intersection_cardinality, expected):
    this_sketch = VectorOfCounts(num_buckets=4, random_seed=0)
    this_sketch.stats = np.array([2, 2, 0, 0])
    that_sketch = VectorOfCounts(num_buckets=4, random_seed=0)
    that_sketch.stats = np.array([2, 0, 2, 0])
    pairwise_estimator = PairwiseEstimator(clip=True, epsilon=epsilon)
    res = pairwise_estimator._get_std_of_intersection(
        intersection_cardinality, this_sketch, that_sketch)
    self.assertAlmostEqual(res, expected, 2)

  @parameterized.parameters(
      (1, 0, 0, 0),
      (1, 0, 4, -0.632),
      (0.5, 4, 0, 0.222),
      (0.5, 4, 4, 0))
  def test_evaluate_closeness_to_a_value(
      self, epsilon, intersection_cardinality, value_to_compare_with, expected):
    this_sketch = VectorOfCounts(num_buckets=4, random_seed=0)
    this_sketch.stats = np.array([2, 2, 0, 0])
    that_sketch = VectorOfCounts(num_buckets=4, random_seed=0)
    that_sketch.stats = np.array([2, 0, 2, 0])
    pairwise_estimator = PairwiseEstimator(clip=True, epsilon=epsilon)
    res = pairwise_estimator.evaluate_closeness_to_a_value(
        intersection_cardinality, value_to_compare_with,
        this_sketch, that_sketch)
    self.assertAlmostEqual(res, expected, 2)


class SequentialEstimatorTest(absltest.TestCase):

  def test_estimate_cardinality_no_clip(self):
    sketch_list = []
    for _ in range(3):
      sketch = VectorOfCounts(num_buckets=2, random_seed=3)
      sketch.add_ids([1])
      sketch_list.append(sketch)
    estimator = SequentialEstimator()
    result = estimator(sketch_list)[0]
    actual = 1.75
    self.assertEqual(result, actual)

  def test_estimate_cardinality_with_clip(self):
    base_sketch = VectorOfCounts(num_buckets=64, random_seed=3)
    base_sketch.add_ids(range(100))
    sketch_list_a = [base_sketch]
    sketch_list_b = [base_sketch]
    for _ in range(3):
      empty_sketch = VectorOfCounts(num_buckets=64, random_seed=3)
      sketch_list_a.append(empty_sketch)  # add empty sketch
      sketch_list_b.append(base_sketch)  # add same sketch
    estimator = SequentialEstimator(clip=True)
    result_a = estimator(sketch_list_a)[0]
    result_b = estimator(sketch_list_b)[0]
    self.assertEqual(result_a, base_sketch.cardinality(),
                     msg='Fail to detect the no-intersection case.')
    self.assertEqual(result_b, base_sketch.cardinality(),
                     msg='Fail to detect the full-intersection case.')


if __name__ == '__main__':
  absltest.main()
