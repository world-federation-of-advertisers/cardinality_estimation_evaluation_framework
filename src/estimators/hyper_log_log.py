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


class HyperLogLogPlusPlus(SketchBase):
  """HyperLogLogPlusPlus Sketch.

  More accurately referred to as HyperLogLogPartialPlusPlus.  We do not
  implement several of the ++ improvements, namely bias estimation and our
  sparse mode is much simpler at the cost of using extra memory.

  Also technically has to be 64 bits to be ++, but for testing and
  experimentation purposes, we allow changing the number of bits here.

  Paper reference can be found here:
  https://storage.googleapis.com/pub-tools-public-publication-data/pdf/40671.pdf
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
    """Creates a Hyper-Log-Log++ object.

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
    self.buckets = np.zeros(self.vector_length_m, dtype=np.int32)

    self.my_hasher = hash_class(random_seed, 2**num_integer_bits)

    # Int to help get the least-significant bits
    self._w_helper_int = int(
        '0' * self.log2_vector_length_p + '1' * self.num_bucket_bits, 2)
    # Int to get the p most significant bits
    self._idx_helper_int = int(
        '1' * self.log2_vector_length_p + '0' * self.num_bucket_bits, 2)

    # Items for our simpler version of sparse mode:
    self.sparse_mode = True
    self.temp_set = set()

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
      self.temp_set.add(value)
      # Check if we're at the threshold yet for normal mode
      if len(self.temp_set) > self.vector_length_m * 6:
        self.sparse_mode = False
        self.temp_set.clear()

    # Then do normal HLL calculation no matter if sparse mode is on
    # (another simplification we made from the paper at the cost of memory)
    hash_value = self.my_hasher(value)

    bucket_index = self.get_idx_bits(hash_value)
    bucket_value_part = self._compute_rho_value(hash_value)

    # Get the appropriate bucket value:
    current_value = self.buckets[bucket_index]
    self.buckets[bucket_index] = max(current_value, bucket_value_part)

  def _raw_hll_cardinality_estimate(self):
    return self.alpha * self.vector_length_m**2 / np.sum(2.0**(-self.buckets))

  def _linear_counting_estimate(self, arg_1_m, arg_2_v):
    # Args not always m and V in paper, hence the arg_i_x notation
    return arg_1_m * np.log(arg_1_m / arg_2_v)

  def _estimate_bias(self, x):
    _ = self, x
    # Not defined for now...
    return 0.0

  def estimate_cardinality(self):
    """Returns the estimated cardinality of this sketch.

    Note we have decided not to implement the bias estimate for now.
    """
    if self.sparse_mode:
      return len(self.temp_set)

    raw_estimate_e = self._raw_hll_cardinality_estimate()

    if raw_estimate_e <= 5 * self.vector_length_m:
      adjusted_estimate_e_prime = raw_estimate_e - self._estimate_bias(
          raw_estimate_e)
    else:
      adjusted_estimate_e_prime = raw_estimate_e

    count_of_registers_at_zero_v = np.sum(self.buckets == 0)
    if count_of_registers_at_zero_v != 0:
      candidate_output_h = self._linear_counting_estimate(
          self.vector_length_m, count_of_registers_at_zero_v)
    else:
      candidate_output_h = adjusted_estimate_e_prime

    if candidate_output_h <= self._threshold_dict[self.log2_vector_length_p]:
      return candidate_output_h
    else:
      return adjusted_estimate_e_prime

  def merge(self, other_hll):
    """Returns a new HLL++ merged with the input arg.

    Args:
      other_hll: another HLL++ object, must be compatible.

    Raises:
      AssertionError: when HLL++s are incompatible.

    Returns:
      HyperLogLogPlusPlus: merged from self and other_hll
    """
    self.assert_compatible(other_hll)

    output_hll = copy.deepcopy(other_hll)
    output_hll.buckets = np.max([self.buckets, output_hll.buckets], axis=0)

    # Handle sparse mode merging:
    output_hll.sparse_mode = output_hll.sparse_mode and self.sparse_mode
    if output_hll.sparse_mode:
      output_hll.temp_set = output_hll.temp_set.union(self.temp_set)
      # To cover the case where the new HLL shouldn't be in sparse mode, pop a
      # random item from the set and add it back to trigger the check
      if output_hll.temp_set:
        random_item = output_hll.temp_set.pop()
        output_hll.add(random_item)
    else:
      output_hll.temp_set = set()

    return output_hll


class HllCardinality(EstimatorBase):
  """A class that unions HLL++s and estimates the combined cardinality."""

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
    return [HllCardinality.union_sketches(sketch_list).estimate_cardinality()]
