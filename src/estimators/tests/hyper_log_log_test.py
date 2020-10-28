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
"""Tests for bloom_filter.py."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HllCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HyperLogLogPlusPlus


class NoOpHasher():
  """For testing purposes: just returns the input int."""

  def __init__(self, *_):
    pass

  def __call__(self, int_in):
    return int_in


class HyperLogLogPlusPlusTest(absltest.TestCase):
  """Test the HLL++ Sketch construction."""

  def setUp(self):
    super().setUp()
    self.vector_length = 16
    self.num_integer_bits = 8
    # 4 bucket bits
    self.bucket_idx_to_bin_str = {
        0: '0000',
        1: '0010',
        4: '0100',
        6: '0110',
        8: '1000',
        15: '1111',
    }
    # 4 leading zero bits
    self.bin_str_to_leading_zeros = {
        '1111': 0,
        '0100': 1,
        '0010': 2,
        '0110': 1,
        '0001': 3,
        '1000': 0,
        '0000': 4,
    }

  def test_single_correct_bucket_placement(self):
    for bucket_idx, bucket_bin_str in self.bucket_idx_to_bin_str.items():
      for leading_0_bin_str, num_leading_0s in self.bin_str_to_leading_zeros.items(
      ):

        hll = HyperLogLogPlusPlus(
            length=self.vector_length,
            random_seed=42,
            hash_class=NoOpHasher,
            num_integer_bits=self.num_integer_bits)

        total_bin_str = bucket_bin_str + leading_0_bin_str
        hll.add(int(total_bin_str, 2))

        expected_buckets = np.zeros(16, dtype=np.int32)
        expected_buckets[bucket_idx] = num_leading_0s + 1

        self.assertSameElements(hll.buckets, expected_buckets)

  def test_max_operation_correct(self):
    for bucket_idx, bucket_bin_str in self.bucket_idx_to_bin_str.items():
      hll = HyperLogLogPlusPlus(
          length=self.vector_length,
          random_seed=42,
          hash_class=NoOpHasher,
          num_integer_bits=self.num_integer_bits)
      max_seen_leading_zeros = 0
      for leading_0_bin_str, num_leading_0s in self.bin_str_to_leading_zeros.items(
      ):

        max_seen_leading_zeros = max(max_seen_leading_zeros, num_leading_0s)
        total_bin_str = bucket_bin_str + leading_0_bin_str
        hll.add(int(total_bin_str, 2))

        expected_buckets = np.zeros(self.vector_length, dtype=np.int32)
        expected_buckets[bucket_idx] = max_seen_leading_zeros + 1

        self.assertSameElements(hll.buckets, expected_buckets)

  def test_simple_estimate_smaller(self):
    hll = HyperLogLogPlusPlus(
        length=self.vector_length,
        random_seed=42,
        num_integer_bits=self.num_integer_bits)

    one_vector = np.ones(self.vector_length)
    hll.buckets = one_vector
    hll.sparse_mode = False
    alpha_16 = 0.673
    hll_should_estimate = alpha_16 * self.vector_length**2 * 2 / self.vector_length

    self.assertEqual(alpha_16, hll.alpha)
    self.assertEqual(hll.estimate_cardinality(), hll_should_estimate)

  def test_simple_estimate_larger(self):
    m = 2**14
    hll = HyperLogLogPlusPlus(
        length=m,
        random_seed=42,
        num_integer_bits=self.num_integer_bits)

    thirty_vector = 30 * np.ones(m)
    hll.buckets = thirty_vector
    hll.sparse_mode = False
    alpha_m = 0.7213 / (1 + 1.079 / m)
    hll_should_estimate = alpha_m * m**2 * 2**30 / m

    self.assertEqual(alpha_m, hll.alpha)
    self.assertEqual(hll.estimate_cardinality(), hll_should_estimate)

  def test_insert_same(self):
    hll = HyperLogLogPlusPlus(random_seed=42)

    hll.add(1)
    card_one = hll.estimate_cardinality()
    hll.add(1)

    self.assertEqual(card_one, hll.estimate_cardinality())

  def insertion_test_helper(self, number_to_insert, acceptable_error=.05):
    hll = HyperLogLogPlusPlus(random_seed=137)

    for i in range(number_to_insert):
      hll.add(i)

    error_ratio = hll.estimate_cardinality() / number_to_insert
    self.assertAlmostEqual(error_ratio, 1.0, delta=acceptable_error)

  def test_insert_small(self):
    self.insertion_test_helper(10)

  def test_insert_medium(self):
    self.insertion_test_helper(1_000)

  def test_insert_large(self):
    self.insertion_test_helper(100_000)

  def test_insert_huge(self):
    self.insertion_test_helper(1_000_000)

  def test_merge_sparse_with_sparse_to_sparse(self):
    hll1 = HyperLogLogPlusPlus(length=16, random_seed=234)
    hll1.add(1)
    hll2 = HyperLogLogPlusPlus(length=16, random_seed=234)
    hll2.add(1)
    merged_hll = hll1.merge(hll2)
    self.assertTrue(merged_hll.sparse_mode,
                    'Merged sketch is not in sparse mode.')
    self.assertTrue(all(hll1.buckets == merged_hll.buckets),
                    'Merged sketch is not correct.')
    self.assertSameElements(merged_hll.temp_set, set([1]),
                            'Temp set is not correct.')
    self.assertEqual(merged_hll.estimate_cardinality(), 1,
                     'Estimated cardinality is not correct.')

  def test_merge_sparse_with_sparse_to_dense(self):
    hll1 = HyperLogLogPlusPlus(length=16, random_seed=234)
    hll2 = HyperLogLogPlusPlus(length=16, random_seed=234)
    for i in range(int(16 * 6 / 2)):
      hll1.add(i)
      hll2.add(i + 100)

    merged_hll = hll1.merge(hll2)
    self.assertTrue(merged_hll.sparse_mode,
                    'Merged sketch should be in sparse mode.')
    self.assertEqual(merged_hll.estimate_cardinality(), 96,
                     'Estimated cardinality not correct under sparse mode.')

    hll1.add(1000)
    merged_hll = hll1.merge(hll2)
    self.assertFalse(merged_hll.sparse_mode,
                     'Merged sketch should not be in sparse mode.')
    self.assertAlmostEqual(
        merged_hll.estimate_cardinality(), 97, delta=97 * 0.05,
        msg='Estimated cardinality not correct under dense mode.'
    )

  def test_merge_sparse_with_dense(self):
    hll1 = HyperLogLogPlusPlus(length=16, random_seed=234)
    hll1.add(100)
    hll2 = HyperLogLogPlusPlus(length=16, random_seed=234)
    for i in range(16 * 6 + 1):
      hll2.add(i)

    merged_hll = hll1.merge(hll2)
    self.assertFalse(merged_hll.sparse_mode,
                     'Merged sketch should not be in sparse mode.')
    # Should change one bucket value given this random seed.
    self.assertEqual(sum(hll2.buckets == merged_hll.buckets), 16 - 1,
                     'Merged sketch is not correct.')
    self.assertSameElements(merged_hll.temp_set, set(),
                            'Temp set is not correct.')
    self.assertGreater(merged_hll.estimate_cardinality(),
                       hll2.estimate_cardinality())

  def test_merge_dense_with_dense(self):
    hll1 = HyperLogLogPlusPlus(length=16, random_seed=234)
    hll2 = HyperLogLogPlusPlus(length=16, random_seed=234)
    for i in range(16 * 6 + 1):
      hll1.add(i)
      hll2.add(i + 100)

    merged_hll = hll1.merge(hll2)
    self.assertFalse(merged_hll.sparse_mode,
                     'Merged sketch should not be in sparse mode.')
    self.assertGreater(sum(hll2.buckets == merged_hll.buckets), 0,
                       'Merged sketch is not correct.')
    self.assertSameElements(merged_hll.temp_set, set(),
                            'Temp set is not correct.')
    self.assertAlmostEqual(
        merged_hll.estimate_cardinality(), 194, delta=194 * 0.1
        )


class HyperLogLogPlusPlusEstimatorTest(absltest.TestCase):
  """Note this could be more comprehensive."""

  def estimator_tester_helper(self, number_of_hlls, acceptable_error=.05):
    estimator = HllCardinality()
    hll_list = []
    for i in range(number_of_hlls):
      hll = HyperLogLogPlusPlus(random_seed=42)
      hll.add(i)
      hll_list.append(hll)

    error_ratio = estimator(hll_list)[0] / number_of_hlls
    self.assertAlmostEqual(error_ratio, 1.0, delta=acceptable_error)

  def test_estimator_small(self):
    self.estimator_tester_helper(10)

  def test_estimator_medium(self):
    self.estimator_tester_helper(1_000)

  def test_estimator_large(self):
    self.estimator_tester_helper(100_000)

  def test_estimator_cardinality_sparse_mode(self):
    estimator = HllCardinality()
    for truth in [0, 1, 1024]:
      hll = HyperLogLogPlusPlus(random_seed=89, length=1024)
      for i in range(truth):
        hll.add(i)
      estimated = estimator([hll])[0]
      self.assertEqual(estimated, truth)

  def test_estimator_cardinality_dense_mode(self):
    estimator = HllCardinality()
    for truth in [1025, 2048]:
      hll = HyperLogLogPlusPlus(random_seed=89, length=1024)
      for i in range(truth):
        hll.add(i)
      estimated = estimator([hll])[0]
      self.assertAlmostEqual(estimated, truth, delta=truth * 0.05)


if __name__ == '__main__':
  absltest.main()
