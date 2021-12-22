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
"""Tests for freq_log_log.py."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import FINGERPRINT_IDX
from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import FREQUENCY_IDX
from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import LEADING_ZEROS_IDX
from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import FreqLogLogCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import FreqLogLogPlusPlus

MAX_FREQ = 15


class NoOpHasher():
  """For testing purposes: just returns the input int."""

  def __init__(self, *_):
    pass

  def __call__(self, int_in):
    return int_in


class FreqLogLogPlusPlusTest(absltest.TestCase):
  """Test the FreqLogLog++ Sketch construction."""

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

        fll = FreqLogLogPlusPlus(
            length=self.vector_length,
            random_seed=42,
            hash_class=NoOpHasher,
            num_integer_bits=self.num_integer_bits)

        total_bin_str = bucket_bin_str + leading_0_bin_str
        # Add it twice to ensure frequency is incremented
        fll.add(int(total_bin_str, 2))
        fll.add(int(total_bin_str, 2))

        expected_buckets = np.zeros((16, 3), dtype=np.int32)
        expected_buckets[bucket_idx, LEADING_ZEROS_IDX] = num_leading_0s + 1
        expected_buckets[bucket_idx, FINGERPRINT_IDX] = int(total_bin_str, 2)
        expected_buckets[bucket_idx, FREQUENCY_IDX] = 2

        self.assertSameElements(fll.buckets[:, LEADING_ZEROS_IDX], expected_buckets[:, LEADING_ZEROS_IDX])
        self.assertSameElements(fll.buckets[:, FINGERPRINT_IDX], expected_buckets[:, FINGERPRINT_IDX])
        self.assertSameElements(fll.buckets[:, FREQUENCY_IDX], expected_buckets[:, FREQUENCY_IDX])

  def test_frequency_bucket_update(self):
    fll = FreqLogLogPlusPlus(
      length=self.vector_length,
      random_seed=42,
      hash_class=NoOpHasher,
      num_integer_bits=self.num_integer_bits)

    # add first element with known bucket ID and known leading zeros
    bucket_str = "0011"
    bucket = int(bucket_str, 2)
    leading_zeros_str = "0101"
    num_leading_zeros = 1 + 1
    fp_str = bucket_str + leading_zeros_str
    fp = int(fp_str, 2)
    fll.add(fp)

    expected_buckets = np.zeros((self.vector_length, 3))
    expected_buckets[bucket, :] = (num_leading_zeros, fp, 1)
    self.assertSameElements(fll.buckets[:, LEADING_ZEROS_IDX], expected_buckets[:, LEADING_ZEROS_IDX])
    self.assertSameElements(fll.buckets[:, FINGERPRINT_IDX], expected_buckets[:, FINGERPRINT_IDX])
    self.assertSameElements(fll.buckets[:, FREQUENCY_IDX], expected_buckets[:, FREQUENCY_IDX])

    # add it a second time
    fll.add(fp)
    expected_buckets[bucket, :] = (num_leading_zeros, fp, 2)
    self.assertSameElements(fll.buckets[:, LEADING_ZEROS_IDX], expected_buckets[:, LEADING_ZEROS_IDX])
    self.assertSameElements(fll.buckets[:, FINGERPRINT_IDX], expected_buckets[:, FINGERPRINT_IDX])
    self.assertSameElements(fll.buckets[:, FREQUENCY_IDX], expected_buckets[:, FREQUENCY_IDX])

    # add a different one with same bucket ID and less leading zeros
    leading_zeros_str = "1111"
    nop_fp_str = bucket_str + leading_zeros_str
    nop_fp = int(nop_fp_str, 2)
    fll.add(nop_fp)

    expected_buckets[bucket, :] = (num_leading_zeros, fp, 2)
    self.assertSameElements(fll.buckets[:, LEADING_ZEROS_IDX], expected_buckets[:, LEADING_ZEROS_IDX])
    self.assertSameElements(fll.buckets[:, FINGERPRINT_IDX], expected_buckets[:, FINGERPRINT_IDX])
    self.assertSameElements(fll.buckets[:, FREQUENCY_IDX], expected_buckets[:, FREQUENCY_IDX])

    # add a different one with same bucket ID and more leading zeros
    leading_zeros_str = "0010"
    num_leading_zeros = 2 + 1
    fp_str = bucket_str + leading_zeros_str
    fp = int(fp_str, 2)
    fll.add(fp)

    expected_buckets[bucket, :] = (num_leading_zeros, fp, 1)
    self.assertSameElements(fll.buckets[:, LEADING_ZEROS_IDX], expected_buckets[:, LEADING_ZEROS_IDX])
    self.assertSameElements(fll.buckets[:, FINGERPRINT_IDX], expected_buckets[:, FINGERPRINT_IDX])
    self.assertSameElements(fll.buckets[:, FREQUENCY_IDX], expected_buckets[:, FREQUENCY_IDX])

    # add a different one with same bucket ID, the same leading zeros, and larger fingerprint
    leading_zeros_str = "0011"
    num_leading_zeros = 2 + 1
    fp_str = bucket_str + leading_zeros_str
    fp = int(fp_str, 2)
    fll.add(fp)

    expected_buckets[bucket, :] = (num_leading_zeros, fp, 1)
    self.assertSameElements(fll.buckets[:, LEADING_ZEROS_IDX], expected_buckets[:, LEADING_ZEROS_IDX])
    self.assertSameElements(fll.buckets[:, FINGERPRINT_IDX], expected_buckets[:, FINGERPRINT_IDX])
    self.assertSameElements(fll.buckets[:, FREQUENCY_IDX], expected_buckets[:, FREQUENCY_IDX])

  def test_max_operation_correct(self):
    for bucket_idx, bucket_bin_str in self.bucket_idx_to_bin_str.items():
      fll = FreqLogLogPlusPlus(
          length=self.vector_length,
          random_seed=42,
          hash_class=NoOpHasher,
          num_integer_bits=self.num_integer_bits)
      max_seen_leading_zeros = 0
      for leading_0_bin_str, num_leading_0s in self.bin_str_to_leading_zeros.items(
      ):

        max_seen_leading_zeros = max(max_seen_leading_zeros, num_leading_0s)
        total_bin_str = bucket_bin_str + leading_0_bin_str
        fll.add(int(total_bin_str, 2))

        expected_buckets = np.zeros((self.vector_length, 3), dtype=np.int32)
        expected_buckets[bucket_idx, 0] = max_seen_leading_zeros + 1

        self.assertSameElements(fll.buckets[:, 0], expected_buckets[:, 0])

  def test_simple_estimate_smaller(self):
    fll = FreqLogLogPlusPlus(
        length=self.vector_length,
        random_seed=42,
        num_integer_bits=self.num_integer_bits)

    one_vector = np.ones((self.vector_length, 3))
    fll.buckets = one_vector
    freq_vector = np.zeros(self.vector_length)
    for i in range(self.vector_length):
      freq_vector[i] = i

    fll.buckets[:, FREQUENCY_IDX] = freq_vector
    fll.sparse_mode = False
    alpha_16 = 0.673
    fll_should_estimate = alpha_16 * self.vector_length**2 * 2 / self.vector_length

    self.assertEqual(alpha_16, fll.alpha)

    freq_dist = np.zeros(MAX_FREQ)
    for i in range(self.vector_length):
      if i < MAX_FREQ:
        freq_dist[i] += 1
      else:
        freq_dist[MAX_FREQ-1] += 1
    freq_dist = freq_dist/sum(freq_dist)
    for i in range(MAX_FREQ):
      freq_dist[i] = sum(freq_dist[i:])
    expected = [round(x) for x in fll_should_estimate*freq_dist]
    self.assertEqual(fll.estimate_cardinality(), expected)

  def test_simple_estimate_larger(self):
    m = 2**14
    fll = FreqLogLogPlusPlus(
        length=m,
        random_seed=42,
        num_integer_bits=self.num_integer_bits)

    # frequency is 1 for all buckets and fingerprint is one
    # we set leading zeros buckets to 30 for the test
    thirty_vector = np.ones((m, 3))
    thirty_vector[:, LEADING_ZEROS_IDX] *= 30
    fll.buckets = thirty_vector
    fll.sparse_mode = False
    alpha_m = 0.7213 / (1 + 1.079 / m)
    fll_should_estimate = alpha_m * m**2 * 2**30 / m

    self.assertEqual(alpha_m, fll.alpha)
    expected = [round(fll_should_estimate)] + [0]*(MAX_FREQ-1)
    self.assertEqual(fll.estimate_cardinality(), expected)

  def test_insert_same(self):
    fll = FreqLogLogPlusPlus(random_seed=42)

    fll.add(1)
    card_one = fll.estimate_cardinality()
    fll.add(1)

    # These should no be equal. First has 0th index second.
    # Second has 1st set.
    self.assertNotEqual(card_one, fll.estimate_cardinality())

  def insertion_test_helper(self, number_to_insert, acceptable_error=.05):
    fll = FreqLogLogPlusPlus(random_seed=137)

    for i in range(number_to_insert):
      fll.add(i)

    error_ratio = fll.estimate_cardinality()[0] / number_to_insert
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
    fll1 = FreqLogLogPlusPlus(length=16, random_seed=234)
    fll1.add(1)
    fll2 = FreqLogLogPlusPlus(length=16, random_seed=234)
    fll2.add(1)

    # merged sketch should have 2+ cardinality = 1
    merged_fll = fll1.merge(fll2)
    self.assertTrue(merged_fll.sparse_mode,
                    'Merged sketch is not in sparse mode.')
    self.assertTrue(all(fll1.buckets[:, 0] == merged_fll.buckets[:, 0]),
                    'Merged sketch is not correct.')
    self.assertSameElements(merged_fll.temp_set, set([1]),
                            'Temp set is not correct.')
    expected = [1, 1] + [0] * (MAX_FREQ-2)
    self.assertEqual(merged_fll.estimate_cardinality(), expected,
                     'Estimated cardinality is not correct.')

  def test_merge_sparse_with_sparse_to_dense(self):
    fll1 = FreqLogLogPlusPlus(length=16, random_seed=234)
    fll2 = FreqLogLogPlusPlus(length=16, random_seed=234)
    for i in range(int(16 * 6 / 2)):
      fll1.add(i)
      fll2.add(i + 100)

    merged_fll = fll1.merge(fll2)
    self.assertTrue(merged_fll.sparse_mode,
                    'Merged sketch should be in sparse mode.')
    expected = [96] + [0] * (MAX_FREQ-1)
    self.assertEqual(merged_fll.estimate_cardinality(), expected,
                     'Estimated cardinality not correct under sparse mode.')

    fll1.add(1000)
    merged_fll = fll1.merge(fll2)
    self.assertFalse(merged_fll.sparse_mode,
                     'Merged sketch should not be in sparse mode.')
    self.assertAlmostEqual(
        merged_fll.estimate_cardinality_float()[0], 97, delta=97 * 0.05,
        msg='Estimated cardinality not correct under dense mode.'
    )

  def test_merge_sparse_with_dense(self):
    fll1 = FreqLogLogPlusPlus(length=16, random_seed=234)
    fll1.add(100)
    fll2 = FreqLogLogPlusPlus(length=16, random_seed=234)
    for i in range(16 * 6 + 1):
      fll2.add(i)

    merged_fll = fll1.merge(fll2)
    self.assertFalse(merged_fll.sparse_mode,
                     'Merged sketch should not be in sparse mode.')
    # Should change one bucket value given this random seed.
    self.assertEqual(sum(fll2.buckets[:, 0] == merged_fll.buckets[:, 0]), 16 - 1,
                     'Merged sketch is not correct.')
    self.assertSameElements(merged_fll.temp_set, set(),
                            'Temp set is not correct.')
    self.assertGreater(merged_fll.estimate_cardinality(),
                       fll2.estimate_cardinality())

  def test_merge_dense_with_dense(self):
    fll1 = FreqLogLogPlusPlus(length=16, random_seed=234)
    fll2 = FreqLogLogPlusPlus(length=16, random_seed=234)
    for i in range(16 * 6 + 1):
      fll1.add(i)
      fll2.add(i + 100)

    merged_fll = fll1.merge(fll2)
    self.assertFalse(merged_fll.sparse_mode,
                     'Merged sketch should not be in sparse mode.')
    self.assertGreater(sum(fll2.buckets[:, 0] == merged_fll.buckets[:, 0]), 0,
                       'Merged sketch is not correct.')
    self.assertSameElements(merged_fll.temp_set, set(),
                            'Temp set is not correct.')
    self.assertAlmostEqual(
        merged_fll.estimate_cardinality()[0], 194, delta=194 * 0.1
        )

  def test_rolling_frequency_estimate_sparse(self):
    fll = FreqLogLogPlusPlus(length=16, random_seed=234)
    x = 100
    for i in range(MAX_FREQ+1):
      fll.add(x)
      desired_index = i if i < MAX_FREQ else MAX_FREQ-1
      self.assertEqual(fll.estimate_cardinality()[desired_index], 1)
    self.assertTrue(fll.sparse_mode)

  def test_rolling_frequency_merge_estimate_sparse(self):
    merged = FreqLogLogPlusPlus(length=16, random_seed=234)
    x = 100
    expected_frequency = [0] * MAX_FREQ
    # For each loop we are incresing "(i+1)" frequency by 1
    # for all values less than or equal to i.
    for i in range(MAX_FREQ):
      working = FreqLogLogPlusPlus(length=16, random_seed=234)
      working.add_ids([x+i]*(i+1))
      merged = merged.merge(working)
      for j in range(i+1):
        expected_frequency[j] += 1

      self.assertEqual(
        merged.estimate_cardinality(), expected_frequency,
        "cardinality at iteration %s is incorrect" % i)
    self.assertTrue(merged.sparse_mode)

  def test_frequency_estimate_non_sparse(self):
    pass


class FreqLogLogPlusPlusEstimatorTest(absltest.TestCase):
  """Note this could be more comprehensive."""

  def estimator_tester_helper(self, number_of_flls, acceptable_error=.05):
    estimator = FreqLogLogCardinality()
    fll_list = []
    for i in range(number_of_flls):
      fll = FreqLogLogPlusPlus(random_seed=42)
      fll.add(i)
      fll_list.append(fll)

    error_ratio = estimator(fll_list)[0] / number_of_flls
    self.assertAlmostEqual(error_ratio, 1.0, delta=acceptable_error)

  def test_estimator_small(self):
    self.estimator_tester_helper(10)

  def test_estimator_medium(self):
    self.estimator_tester_helper(1_000)

  #def test_estimator_large(self):
  #  self.estimator_tester_helper(10_000)

  def test_estimator_cardinality_sparse_mode(self):
    estimator = FreqLogLogCardinality()
    for truth in [0, 1, 1024]:
      fll = FreqLogLogPlusPlus(random_seed=89, length=1024)
      for i in range(truth):
        fll.add(i)
      estimated = estimator([fll])
      self.assertEqual(estimated[0], truth)

  def test_estimator_cardinality_dense_mode(self):
    estimator = FreqLogLogCardinality()
    for truth in [1025, 2048]:
      fll = FreqLogLogPlusPlus(random_seed=89, length=1024)
      for i in range(truth):
        fll.add(i)
      estimated = estimator([fll])[0]
      self.assertAlmostEqual(estimated, truth, delta=truth * 0.05)


if __name__ == '__main__':
  absltest.main()
