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
"""A freqloglog test simulation."""
import copy
import math

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common.hash_function import HashFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase

DEFAULT_INT_LENGTH_BITS = 64

LEADING_ZEROS_IDX = 0
FINGERPRINT_IDX = 1
FREQUENCY_IDX = 2


class FreqLogLogPlusPlus(SketchBase):
  """FreqLogLogPlusPlus Sketch.

  This implementation was copied from HyperLogLogPlusPlus in the same directory and
  the frequency extensions were added. This was done to avoid breaking any downstream
  dependencies of HyperLogLogPlusPlus.

  FreqLogLog paper reference can be found here:
  https://storage.googleapis.com/pub-tools-public-publication-data/pdf/54a28925b11e05b1d8d1cc5c03f171666dc77e8e.pdf
  """
  # Static content, used in the cardinality estimation function below
  _threshold_dict = {
      4: 10,
      5: 20,
      6: 40,
      7: 80,
      8: 220,
      9: 400,
      10: 900,
      11: 1800,
      12: 3100,
      13: 6500,
      14: 11500,
      15: 20000,
      16: 50000,
      17: 120000,
      18: 350000,
  }

  @classmethod
  def get_sketch_factory(cls, length):

    def f(random_seed):
      return cls(random_seed, length)

    return f

  def __init__(self,
               random_seed,
               length=16384,
               num_integer_bits=DEFAULT_INT_LENGTH_BITS,
               hash_class=HashFunction):
    """Creates a Freq-Log-Log++ object.

    Note: default length of 16384 = 2**14

    Args:
      random_seed: a random seed for the hash function.
      length: length of the sketch, must be a power of 2.
      num_integer_bits: Number of bits to consider an integer have
      hash_class: a HashFunction object to apply to input ids

    Raises:
      ValueError: for invalid input parameters
    """
    self.log2_vector_length_p = round(math.log2(length))
    self.vector_length_m = 2**self.log2_vector_length_p
    if self.vector_length_m != length:
      raise ValueError(
          f'length of HLL must be power of 2; {length} is an invalid choice')
    if not (self.log2_vector_length_p > 3 and self.log2_vector_length_p < 17):
      raise ValueError(f'Invalid vector length of {self.vector_length_m}')

    if self.vector_length_m < 16:
      raise ValueError(f'length of {length} is too low; try a higher value')
    if self.vector_length_m == 16:
      alpha = 0.673
    elif self.vector_length_m == 32:
      alpha = 0.697
    elif self.vector_length_m == 64:
      alpha = 0.709
    else:
      alpha = 0.7213 / (1 + 1.079 / self.vector_length_m)
    self.alpha = alpha

    # Number of bits left over after using the index bits
    # We use a 64-bit integer output from the hash function
    self.num_integer_bits = num_integer_bits
    # The hash function gets split up into two pieces
    # self.log2_vector_length_p bits go towards the bucket index
    # The remainder go into the value piece, from which we get the number of
    # leading zeros
    self.num_bucket_bits = num_integer_bits - self.log2_vector_length_p

    # Allocate the initial bucket-values which are all zero by definition
    # The first dimension is the number of leading zeroes, which is used
    # by HLL++
    # The second dimension is the fingerprint of the last item
    # that caused a modification of the first dimension. This value
    # is replaced anytime the first dimension is modified.
    # The third dimension is the number of times the fingerprint has
    # been seen.
    self.buckets = np.zeros((self.vector_length_m, 3), dtype=np.int32)

    self.my_hasher = hash_class(random_seed, 2**num_integer_bits)

    # Int to help get the least-significant bits
    self._w_helper_int = int(
        '0' * self.log2_vector_length_p + '1' * self.num_bucket_bits, 2)
    # Int to get the p most significant bits
    self._idx_helper_int = int(
        '1' * self.log2_vector_length_p + '0' * self.num_bucket_bits, 2)

    # Items for our simpler version of sparse mode:
    self.sparse_mode = True
    self.temp_set = {}

    # Max frequency to estimate
    self.max_freq = 15

  def assert_compatible(self, other):
    """Performs check on other to make sure its compatible with self."""
    assert self.log2_vector_length_p == other.log2_vector_length_p, ('Vectors '
                                                                     'not same '
                                                                     'length')
    assert self.my_hasher == other.my_hasher, "Vectors don't have same hash"

  def get_idx_bits(self, x):
    # The 'p' largest bits decide which bucket x goes into
    return (x & self._idx_helper_int) >> self.num_bucket_bits

  def get_w_bits(self, x):
    # The 'INT_BITS - p' smallest bits decide the value of the bucket
    return x & self._w_helper_int

  def _count_leading_zeros(self, int_in):
    most_significant_bit_idx = math.floor(math.log2(int_in))
    return self.num_bucket_bits - most_significant_bit_idx - 1

  def _compute_rho_value(self, hashed_value):
    # number of leading zeros + 1 in the bits that come after the p index bits
    w = self.get_w_bits(hashed_value)
    if w == 0:
      # This means all the bits are zero, return total number of bits in bucket
      return self.num_bucket_bits + 1
    return self._count_leading_zeros(w) + 1

  def add(self, value):
    # Handle sparse mode first if needed
    if self.sparse_mode:
      self.temp_set[value] = self.temp_set.get(value, 0) + 1
      # Check if we're at the threshold yet for normal mode
      if len(self.temp_set) > self.vector_length_m * 6:
        self.sparse_mode = False
        self.temp_set.clear()

    # Then do normal HLL calculation no matter if sparse mode is on
    # (another simplification we made from the paper at the cost of memory)
    hash_value = self.my_hasher(value)

    bucket_index = self.get_idx_bits(hash_value)
    bucket_value_part = self._compute_rho_value(hash_value)
    fingerprint_value = hash_value % 2**32

    # Get the appropriate bucket value:
    current_value = self.buckets[bucket_index, LEADING_ZEROS_IDX]
    current_fingerprint = self.buckets[bucket_index, FINGERPRINT_IDX]
    # if new value is bigger then reset the fingerprint and frequency
    # or the bucket values are the same the fingerprint is bigger
    # then reset the fp and the frequency
    if (bucket_value_part > current_value or
        bucket_value_part == current_value and fingerprint_value > current_fingerprint):
      self.buckets[bucket_index, :] = [bucket_value_part, fingerprint_value, 1]
    # if hashes are the same then increment frequency
    elif current_fingerprint == fingerprint_value:
      self.buckets[bucket_index, FREQUENCY_IDX] += 1

  def _raw_hll_cardinality_estimate(self):
    return self.alpha * self.vector_length_m**2 / np.sum(2.0**(-self.buckets[:, LEADING_ZEROS_IDX]))

  def _linear_counting_estimate(self, arg_1_m, arg_2_v):
    # Args not always m and V in paper, hence the arg_i_x notation
    return arg_1_m * np.log(arg_1_m / arg_2_v)

  def _estimate_bias(self, x):
    _ = self, x
    # Not defined for now...
    return 0.0

  def estimate_cardinality_float(self):
    """Returns k+ reach for k <= 15 as possibly floating point numbers.

    Note we have decided not to implement the bias estimate for now.
    """
    frequencies = np.zeros(self.max_freq, dtype=np.int32)
    cardinality = np.zeros(self.max_freq, dtype=np.int32)

    if self.sparse_mode:
      cardinality =  len(self.temp_set)
      for _, freq in self.temp_set.items():
        if freq > self.max_freq:
          freq = self.max_freq
        frequencies[freq-1] += 1
    else:
      raw_estimate_e = self._raw_hll_cardinality_estimate()

      if raw_estimate_e <= 5 * self.vector_length_m:
        adjusted_estimate_e_prime = raw_estimate_e - self._estimate_bias(
          raw_estimate_e)
      else:
        adjusted_estimate_e_prime = raw_estimate_e

      count_of_registers_at_zero_v = np.sum(self.buckets[:, LEADING_ZEROS_IDX] == 0)
      if count_of_registers_at_zero_v != 0:
        candidate_output_h = self._linear_counting_estimate(
          self.vector_length_m, count_of_registers_at_zero_v)
      else:
        candidate_output_h = adjusted_estimate_e_prime

      if candidate_output_h <= self._threshold_dict[self.log2_vector_length_p]:
        cardinality = candidate_output_h
      else:
        cardinality = adjusted_estimate_e_prime

      for freq in self.buckets[:, FREQUENCY_IDX]:
        if freq > self.max_freq:
          freq = self.max_freq
        frequencies[int(freq-1)] += 1

    sum_freqs = sum(frequencies)
    if sum_freqs == 0:
      return [0] * self.max_freq
    freq_dist = frequencies / sum(frequencies)
    # set up the freq_dist to provide k+ reach
    freq_dist[0] = 1
    for i in range(1, len(freq_dist)):
      freq_dist[i] = sum(freq_dist[i:])

    return [x for x in cardinality * freq_dist]

  def estimate_cardinality(self):
    """Returns k+ reach for k <= 15

    Note we have decided not to implement the bias estimate for now.
    """
    return [round(x) for x in self.estimate_cardinality_float()]

  def merge(self, other_fll):
    """Returns a new FLL++ merged with the input arg.

    Args:
      other_fll: another HLL++ object, must be compatible.

    Raises:
      AssertionError: when HLL++s are incompatible.

    Returns:
      HyperLogLogPlusPlus: merged from self and other_fll
    """
    self.assert_compatible(other_fll)

    output_fll = copy.deepcopy(other_fll)
    for i in range(self.vector_length_m):
      # if self is greater then replace output with that
      if self.buckets[i, LEADING_ZEROS_IDX] > output_fll.buckets[i, LEADING_ZEROS_IDX]:
        output_fll.buckets[i, :] = self.buckets[i, :]
      # if they have the same hash values, sum the frequencies
      elif self.buckets[i, FINGERPRINT_IDX] == output_fll.buckets[i, FINGERPRINT_IDX]:
        output_fll.buckets[i, FREQUENCY_IDX] += self.buckets[i, FREQUENCY_IDX]

    # Handle sparse mode merging:
    output_fll.sparse_mode = output_fll.sparse_mode and self.sparse_mode
    if output_fll.sparse_mode:
      for value, freq in self.temp_set.items():
        output_fll.temp_set[value] = output_fll.temp_set.get(value, 0) + freq
      # See if we should still be in sparse mode.
      if len(output_fll.temp_set) > self.vector_length_m * 6:
        output_fll.temp_set.clear()
        output_fll.sparse_mode = False
    else:
      output_fll.temp_set = {}

    return output_fll


class FreqLogLogCardinality(EstimatorBase):
  """A class that unions FLL++s and estimates the combined cardinality."""

  def __init__(self):
    EstimatorBase.__init__(self)

  @staticmethod
  def union_sketches(sketch_list):
    working_sketch = sketch_list[0]
    for s in sketch_list[1:]:
      working_sketch = working_sketch.merge(s)
    return working_sketch

  def __call__(self, sketch_list):
    """Does a bit-wise of all sketches and returns a combined cardinality estimate."""
    return FreqLogLogCardinality.union_sketches(sketch_list).estimate_cardinality()
