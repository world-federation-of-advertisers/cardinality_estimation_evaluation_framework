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

"""Tests for hash.py."""

from absl.testing import absltest

from wfa_cardinality_estimation_evaluation_framework.common.hash_function import HashFunction


class HashTest(absltest.TestCase):

  def test_inequality_seed(self):
    hash_a = HashFunction(1, 5)
    hash_b = HashFunction(2, 5)

    self.assertNotEqual(hash_a, hash_b)

  def test_inequality_mod(self):
    hash_a = HashFunction(1, 5)
    hash_b = HashFunction(1, 3)

    self.assertNotEqual(hash_a, hash_b)

  def test_equality(self):
    hash_a = HashFunction(1, 3)
    hash_b = HashFunction(1, 3)

    self.assertEqual(hash_a, hash_b)

  def test_hashing(self):
    hash_a = HashFunction(1, 3)
    hash_b = HashFunction(1, 3)
    x = 137
    self.assertEqual(hash_a(x), hash_b(x))

  def test_ordering(self):
    hash_a = HashFunction(1, 3)
    hash_b = HashFunction(1, 5)
    hash_c = HashFunction(1, 5)
    hash_d = HashFunction(2, 5)

    self.assertLess(hash_a, hash_b)
    self.assertLess(hash_c, hash_d)

  def test_modulus_too_big(self):
    self.assertRaises(ValueError, lambda: HashFunction(42, 2**65))


if __name__ == '__main__':
  absltest.main()
