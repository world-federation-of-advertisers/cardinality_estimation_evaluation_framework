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

"""Tests for any_sketch.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import AnySketch
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import BitwiseOrFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import GeometricDistribution
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import IndexSpecification
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import LogBucketDistribution
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import SketchConfig
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import SumFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import UniformDistribution


class DistributionTest(parameterized.TestCase):

  def test_uniform_distribution(self):
    u1 = UniformDistribution(10)
    u2 = UniformDistribution(10)
    self.assertLen(u1, 10)
    self.assertEqual(u1, u2)
    self.assertEqual(u1.get_index(42), 2)

  def test_geo_bucket_distribution_get_bounds(self):
    bounds = GeometricDistribution(num_values=4, probability=0.08).register_bounds
    expected = np.array([0.28208044, 0.54159445, 0.78034734, 1.])
    np.testing.assert_allclose(bounds, expected, atol=1e-8)

  @parameterized.parameters(
      (0, 0),
      (0.28208044, 0),
      (0.28208045, 1),
      (0.54159445, 1),
      (0.5415946, 2),
      (0.94, 3),
      (1, 3))
  def test_geo_bucket_distribution_get_index(self, hash_value, index):
    exp_dist = GeometricDistribution(num_values=4, probability=0.08)
    self.assertEqual(exp_dist.get_index(hash_value, max_hash_value=1), index)

  def test_log_bucket_distribution_get_bounds(self):
    bounds = LogBucketDistribution(num_values=4).register_bounds
    expected = np.array([0.49373838, 0.77483521, 0.93154471, 1])
    np.testing.assert_allclose(bounds, expected, atol=1e-8)

  @parameterized.parameters(
      (0, 0),
      (0.49373838, 1),
      (0.5, 1),
      (0.8, 2),
      (0.94, 3),
      (1, 3))
  def test_log_bucket_distribution_get_index(self, hash_value, index):
    exp_dist = LogBucketDistribution(num_values=4)
    self.assertEqual(exp_dist.get_index(hash_value, max_hash_value=1), index)


class AnySketchTest(absltest.TestCase):

  def test_basic_capabilities(self):
    s = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=1,
                     value_functions=[SumFunction()]))
    s.add_ids([1, 1])
    self.assertIn(1, s)
    self.assertEqual(s.max_size(), 10)
    self.assertTrue(s.assert_compatible(s))

  def test_incompatible_different_hashes(self):
    # S1 and S2 will have default and therefore different random seeds.
    s1 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=2,
                     value_functions=[SumFunction()]))
    s2 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=2,
                     value_functions=[SumFunction()]))
    with self.assertRaises(AssertionError):
      s1.assert_compatible(s2)

  def test_incompatible_different_configs_distribution(self):
    # S1 and S2 have different distribution sketch in config.
    s1 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=1,
                     value_functions=[SumFunction()]), 42)
    s2 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(11), 'uniform')],
                     num_hashes=1,
                     value_functions=[SumFunction()]), 42)
    with self.assertRaises(AssertionError):
      s1.assert_compatible(s2)

  def test_incompatible_different_configs_value_function(self):
    # S1 and S2 have different value function in sketch config.
    s1 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=1,
                     value_functions=[SumFunction()]), 42)
    s2 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=1,
                     value_functions=[BitwiseOrFunction()]), 42)
    with self.assertRaises(AssertionError):
      s1.assert_compatible(s2)

  def test_run_through_compatible_sketches(self):
    # S1 and S2 are compatible.
    s1 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=1,
                     value_functions=[SumFunction()]), 42)
    s2 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(10), 'uniform')],
                     num_hashes=1,
                     value_functions=[SumFunction()]), 42)
    self.assertTrue(
        s1.assert_compatible(s2),
        'Two sketches are compatible, yet raises error.')

  def test_with_custom_hash_function(self):

    class TestHash(object):

      def __init__(self, seed):
        self.seed = seed

      def __call__(self, x):
        return 1

    s1 = AnySketch(
        SketchConfig([IndexSpecification(UniformDistribution(2), 'uniform'),
                      IndexSpecification(UniformDistribution(4), 'uniform')],
                     num_hashes=3,
                     value_functions=[SumFunction()]), 42, TestHash)
    s1.add(1)
    expected = np.array([[0, 0, 0, 0], [0, 3, 0, 0]])
    self.assertLen(s1.sketch, len(expected))
    for i, sketch in enumerate(expected):
      np.testing.assert_equal(s1.sketch[i], sketch)


class ValueFunctionTest(parameterized.TestCase):

  def test_sum(self):
    sum_func = SumFunction()
    self.assertEqual(sum_func(1, 1), 2, 'SumFunction is not correct')

  @parameterized.parameters(
      (1.1, 0),
      (0, 1.1))
  def test_bitwise_or_raise(self, x, y):
    bitwise_or = BitwiseOrFunction()
    with self.assertRaises(AssertionError):
      bitwise_or(x, y)

  @parameterized.parameters(
      (1, 1, 1),
      (0, 0, 0),
      (1, 0, 1),
      (0, 1, 1))
  def test_bitwise_or_works(self, x, y, expected):
    bitwise_or = BitwiseOrFunction()
    self.assertEqual(bitwise_or(x, y), expected, f'{x}^{y} != {expected}')


if __name__ == '__main__':
  absltest.main()
