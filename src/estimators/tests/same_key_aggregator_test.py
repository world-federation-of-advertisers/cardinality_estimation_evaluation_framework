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
"""Tests for wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import UniqueKeyFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import ExponentialSameKeyAggregator
from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import StandardizedHistogramEstimator


class ExponentialSameKeyAggregatorTest(absltest.TestCase):

  def test_add(self):
    expected_exponential_bloom_filter = np.zeros(4, dtype=np.int32)
    expected_unique_key_tracker = np.zeros(4, dtype=np.int32)
    expected_frequency_count_tracker = np.zeros(4, dtype=np.int32)
    ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=10, random_seed=0)
    ska.add(0)

    expected_exponential_bloom_filter[0] = 1
    expected_unique_key_tracker[0] = 1
    np.testing.assert_equal(
        ska.unique_key_tracker.sketch, expected_unique_key_tracker)
    expected_frequency_count_tracker[0] = 1
    np.testing.assert_equal(
        ska.frequency_count_tracker.sketch, expected_frequency_count_tracker)

  def test_add_ids(self):
    expected_exponential_bloom_filter = np.zeros(4, dtype=np.int32)
    expected_unique_key_tracker = np.zeros(4, dtype=np.int32)
    expected_frequency_count_tracker = np.zeros(4, dtype=np.int32)
    ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=10, random_seed=0)
    ska.add_ids([0, 1])

    expected_exponential_bloom_filter[0] = 1
    np.testing.assert_equal(
        ska.exponential_bloom_filter.sketch, expected_exponential_bloom_filter)
    expected_unique_key_tracker[0] = UniqueKeyFunction.FLAG_COLLIDED_REGISTER
    np.testing.assert_equal(
        ska.unique_key_tracker.sketch, expected_unique_key_tracker)
    expected_frequency_count_tracker[0] = 2
    np.testing.assert_equal(
        ska.frequency_count_tracker.sketch, expected_frequency_count_tracker)

  def test_assert_compatible(self):
    this_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=10, random_seed=0)
    other_ska = ExponentialSameKeyAggregator(
        length=8, decay_rate=10, random_seed=0)
    with self.assertRaises(AssertionError):
      this_ska.assert_compatible(other_ska)


class StandardizedHistogramEstimatorTest(absltest.TestCase):

  def test_merge_two_sketches(self):
    this_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    this_ska.add_ids([0, 1, 3])
    that_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    that_ska.add_ids([1, 1, 2])
    estimator = StandardizedHistogramEstimator()
    merged_ska = estimator.merge_two_sketches(this_ska, that_ska)
    expected_bf_sktech = np.array([1, 0, 1, 1])
    expected_unique_key_sketch = np.array(
        [UniqueKeyFunction.FLAG_COLLIDED_REGISTER,
         UniqueKeyFunction.FLAG_EMPTY_REGISTER,
         3, 2])
    expected_freq_count_sketch = np.array([2, 0, 1, 3])
    np.testing.assert_equal(
        merged_ska.exponential_bloom_filter.sketch,
        expected_bf_sktech,
        err_msg='Merged SameKeyAggregator does not have exp-BF as expected.')
    np.testing.assert_equal(
        merged_ska.unique_key_tracker.sketch,
        expected_unique_key_sketch,
        err_msg='Merged SameKeyAggregator does not have unique keys'
        'as expected.')
    np.testing.assert_equal(
        merged_ska.frequency_count_tracker.sketch,
        expected_freq_count_sketch,
        err_msg='Merged SameKeyAggregator does not have frequency counts'
        'as expected.')

  def test_merge_sketch_list(self):
    first_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    first_ska.add_ids([0, 1, 3])
    second_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    second_ska.add_ids([1, 1, 2])
    third_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    third_ska.add_ids([1, 1, 4, 4])
    estimator = StandardizedHistogramEstimator()
    merged_ska = estimator.merge_sketch_list(
        [first_ska, second_ska, third_ska])
    expected_bf_sktech = np.array([1, 0, 1, 1])
    expected_unique_key_sketch = np.array(
        [UniqueKeyFunction.FLAG_COLLIDED_REGISTER,
         UniqueKeyFunction.FLAG_EMPTY_REGISTER,
         3, 2])
    expected_freq_count_sketch = np.array([4, 0, 1, 5])
    np.testing.assert_equal(
        merged_ska.exponential_bloom_filter.sketch,
        expected_bf_sktech,
        err_msg='Merged SameKeyAggregator does not have exp-BF as expected.')
    np.testing.assert_equal(
        merged_ska.unique_key_tracker.sketch,
        expected_unique_key_sketch,
        err_msg='Merged SameKeyAggregator does not have unique keys'
        'as expected.')
    np.testing.assert_equal(
        merged_ska.frequency_count_tracker.sketch,
        expected_freq_count_sketch,
        err_msg='Merged SameKeyAggregator does not have frequency counts'
        'as expected.')

  def test_estimate_one_plus_reach(self):
    ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    ska.add_ids([0, 1, 2, 3])
    estimator = StandardizedHistogramEstimator(noiser_class=None)
    cardinality = estimator.estimate_one_plus_reach(ska)
    expected_cardinality = 5.877
    self.assertAlmostEqual(cardinality, expected_cardinality, delta=0.001)

  def test_estimate_histogram_from_effective_keys(self):
    ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    ska.add_ids([0, 1, 1, 1, 1, 2, 2, 3, 6, 6])
    estimator = StandardizedHistogramEstimator(
        max_freq=5, noiser_class=None)
    freq_hist = estimator.estimate_histogram_from_effective_keys(ska)
    expected_freq_hist = np.array([0, 2, 0, 1, 0])
    np.testing.assert_equal(freq_hist, expected_freq_hist)

  def test_standardize_histogram(self):
    histogram = np.array([1, 1, 5, 3, 0])
    estimator = StandardizedHistogramEstimator()
    total = 20
    res = estimator.standardize_histogram(histogram, total)
    expected = np.array([2, 2, 10, 6, 0])
    np.testing.assert_equal(res, expected)

  def test_end_to_end_estimator(self):
    first_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    first_ska.add_ids([0, 1, 3])
    second_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    second_ska.add_ids([1, 1, 2])
    third_ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=1, random_seed=0)
    third_ska.add_ids([1, 1, 4, 4])
    estimator = StandardizedHistogramEstimator(
        max_freq=6, noiser_class=None)
    res = estimator([first_ska, second_ska, third_ska])
    expected = [5.877, 2.939, 2.939, 2.939, 2.939, 0]
    np.testing.assert_almost_equal(res, expected, decimal=3)


if __name__ == '__main__':
  absltest.main()
