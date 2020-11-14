# Lint as: python3
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

import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BlipNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import ExponentialBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FixedProbabilityBitFlipNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import GeometricBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import get_probability_of_flip
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import invert_monotonic
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import SurrealDenoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UniformBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UniformCountingBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UnionEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import GeometricEstimateNoiser


class BloomFilterTest(absltest.TestCase):

  def test_insert_lookup_with_random_seed(self):
    b = BloomFilter(length=15, random_seed=2)

    self.assertNotIn(1, b)
    b.add(1)
    self.assertIn(1, b)
    self.assertNotIn(2, b)

  def test_factory(self):
    factory = BloomFilter.get_sketch_factory(length=15)

    b = factory(2)
    b.add(1)

    self.assertIn(1, b)
    self.assertNotIn(2, b)

  def test_not_compatible_different_seeds(self):
    b1 = BloomFilter(length=15, random_seed=1)
    b2 = BloomFilter(length=15, random_seed=2)
    with self.assertRaises(AssertionError):
      b1.assert_compatible(b2)
    with self.assertRaises(AssertionError):
      b2.assert_compatible(b1)

  def test_not_compatible_different_lengths(self):
    b1 = BloomFilter(length=10, random_seed=2)
    b2 = BloomFilter(length=15, random_seed=2)
    with self.assertRaises(AssertionError):
      b1.assert_compatible(b2)
    with self.assertRaises(AssertionError):
      b2.assert_compatible(b1)

  def test_compatible(self):
    b1 = BloomFilter(length=15, random_seed=2)
    b2 = BloomFilter(length=15, random_seed=2)
    self.assertTrue(b1.assert_compatible(b2))
    self.assertTrue(b2.assert_compatible(b1))


class AnyDistributionBloomFilterTest(parameterized.TestCase):

  @parameterized.parameters(
      (UniformBloomFilter, {}),
      (LogarithmicBloomFilter, {}),
      (ExponentialBloomFilter, {'decay_rate': 1}),
      (GeometricBloomFilter, {'probability': 0.08}),
  )
  def test_insert(self, bloom_filter_class, kwargs):
    adbf = bloom_filter_class(length=2, random_seed=1, **kwargs)
    self.assertEqual(np.sum(adbf.sketch), 0)
    adbf.add([1, 1])
    self.assertEqual(np.sum(adbf.sketch), 1)

  @parameterized.parameters(
      (UniformBloomFilter, {}),
      (LogarithmicBloomFilter, {}),
      (ExponentialBloomFilter, {'decay_rate': 1}),
      (GeometricBloomFilter, {'probability': 0.08}),
  )
  def test_factory(self, bloom_filter_class, kwargs):
    factory = bloom_filter_class.get_sketch_factory(length=4, **kwargs)
    adbf = factory(2)
    self.assertEqual(np.sum(adbf.sketch), 0)
    adbf.add([1, 1])
    self.assertEqual(np.sum(adbf.sketch), 1)

  def test_inversion(self):
    log = invert_monotonic(math.exp, epsilon=0.1**9)
    self.assertAlmostEqual(log(1), 0)
    self.assertAlmostEqual(log(math.exp(2)), 2)
    self.assertAlmostEqual(log(math.exp(42.42)), 42.42)


class GeometricBloomFilterTest(absltest.TestCase):

  def test_insert_lookup_with_random_seed(self):
    b = GeometricBloomFilter(length=15, probability=0.08, random_seed=2)

    self.assertNotIn(1, b)
    b.add(1)
    self.assertIn(1, b)
    self.assertNotIn(2, b)

  def test_factory(self):
    factory = GeometricBloomFilter.get_sketch_factory(length=15,
                                                      probability=0.08)

    b = factory(2)
    b.add(1)

    self.assertIn(1, b)
    self.assertNotIn(2, b)

  def test_not_compatible_different_seeds(self):
    b1 = GeometricBloomFilter(length=15, probability=0.08, random_seed=1)
    b2 = GeometricBloomFilter(length=15, probability=0.08, random_seed=2)
    with self.assertRaises(AssertionError):
      b1.assert_compatible(b2)
    with self.assertRaises(AssertionError):
      b2.assert_compatible(b1)

  def test_not_compatible_different_lengths(self):
    b1 = GeometricBloomFilter(length=10, probability=0.08, random_seed=2)
    b2 = GeometricBloomFilter(length=15, probability=0.08, random_seed=2)
    with self.assertRaises(AssertionError):
      b1.assert_compatible(b2)
    with self.assertRaises(AssertionError):
      b2.assert_compatible(b1)

  def test_compatible(self):
    b1 = GeometricBloomFilter(length=15, probability=0.08, random_seed=2)
    b2 = GeometricBloomFilter(length=15, probability=0.08, random_seed=2)
    self.assertTrue(b1.assert_compatible(b2))
    self.assertTrue(b2.assert_compatible(b1))


class UniformCountingBloomFilterTest(absltest.TestCase):
  def test_insert(self):
    cbf = UniformCountingBloomFilter(length=2, random_seed=1)
    self.assertEqual(np.sum(cbf.sketch), 0)
    cbf.add(1)
    cbf.add(1)
    self.assertEqual(np.sum(cbf.sketch), 2)

  def test_factory(self):
    factory = UniformCountingBloomFilter.get_sketch_factory(length=4)
    cbf = factory(2)
    self.assertEqual(np.sum(cbf.sketch), 0)
    cbf.add(1)
    cbf.add(1)
    self.assertEqual(np.sum(cbf.sketch), 2)


class UnionEstimatorTest(absltest.TestCase):

  def test_single_insertion(self):
    b1 = BloomFilter(length=10, random_seed=3)
    b1.add(1)
    self.assertEqual(UnionEstimator.estimate_cardinality(b1), 1)

  def test_multi_insertion_same_element(self):
    b1 = BloomFilter(length=10, random_seed=3)
    b1.add(1)
    b1.add(1)
    self.assertEqual(UnionEstimator.estimate_cardinality(b1), 1)

  def test_raise_error_with_full_bloom_filter(self):
    b1 = BloomFilter(length=2, random_seed=3)
    b1.add_ids(range(10))
    with self.assertRaises(ValueError):
      _ = UnionEstimator.estimate_cardinality(b1)

  def test_multi_insertion_two_hash(self):
    # Due to hash collisions, the estimate could be 1 or 2.
    # We test the distribution.
    estimates = []
    for i in range(1000):
      b1 = BloomFilter(length=100, num_hashes=2, random_seed=i)
      b1.add(1)
      b1.add(2)
      estimates.append(UnionEstimator.estimate_cardinality(b1))
    self.assertAlmostEqual(np.mean(estimates), 1.941, delta=0.04)

  def test_union_of_two(self):
    b1 = BloomFilter(length=10, random_seed=5)
    b1.sketch[0] = 1

    b2 = BloomFilter(length=10, random_seed=5)
    b2.sketch[1] = 1

    union = UnionEstimator().union_sketches([b1, b2])
    self.assertEqual(UnionEstimator.estimate_cardinality(union), 2)
    expected = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    np.testing.assert_equal(union.sketch, expected,
                            'The union sketch is no correct.')

  def test_cardinality_estimation(self):
    b1 = BloomFilter(length=10, random_seed=3)
    b1.sketch[0] = 1

    b2 = BloomFilter(length=10, random_seed=3)
    b2.sketch[1] = 1

    cardinality = UnionEstimator()([b1, b2])[0]
    self.assertEqual(cardinality, 2)


class FirstMomentEstimatorTest(parameterized.TestCase):

  def test_union_sketches(self):
    adbf_list = [LogarithmicBloomFilter(length=4, random_seed=1)
                 for _ in range(2)]
    adbf_list[0].sketch = np.array([0.1, 0.9, 0.1, 0.9])
    adbf_list[1].sketch = np.array([0.1, 0.1, 0.9, 0.9])
    estimator = FirstMomentEstimator(method='any')
    union = estimator.union_sketches(adbf_list)
    expected = np.array([0.19, 0.91, 0.91, 0.99])
    np.testing.assert_allclose(union.sketch, expected, atol=0.01)
  @parameterized.parameters(
    ([1, 1, 0, 0], 3.38427734375),  # Normal case
    ([2, 0, 1, 1], 0),  # error case (n>=1)
    ([1, 0, 0, 0], 1.0),  # first_moment(lower_bound) > 0
  )
  def test_geo_estimator(self, sketch, expected):
    adbf = GeometricBloomFilter(length=4, probability=0.6, random_seed=1)
    adbf.sketch = np.array(sketch)
    estimator = FirstMomentEstimator(method='geo')
    estimate=estimator([adbf])[0]
    self.assertAlmostEqual(estimate, expected)

  def test_check_compatibility(self):
    # With different random seed.
    adbf_list = [LogarithmicBloomFilter(length=4, random_seed=i)
                 for i in range(2)]
    with self.assertRaises(AssertionError):
      FirstMomentEstimator._check_compatibility(adbf_list)

  @parameterized.parameters(
      (UniformBloomFilter, {}, 'uniform', 1.151),
      (LogarithmicBloomFilter, {}, 'log', 1.333),
      (ExponentialBloomFilter, {'decay_rate': 1}, 'exp', 1.1645),
      (GeometricBloomFilter, {'probability': 0.08}, 'geo', 1.0005),
      (UniformBloomFilter, {}, 'any', 1)
  )
  def test_estimate_cardinality(self, bf, bf_kwargs, method, truth):
    adbf = bf(length=4, random_seed=0, **bf_kwargs)
    adbf.add_ids([1])
    estimator = FirstMomentEstimator(method=method)
    estimate = estimator([adbf])[0]
    self.assertAlmostEqual(estimate, truth, 3, msg=method)

  def test_uniform_bf_corner_cases(self):
    adbf = UniformBloomFilter(length=2, random_seed=0)
    # Test if the sketch is full.
    adbf.sketch = np.array([1, 1])
    estimator = FirstMomentEstimator(method='uniform')
    self.assertTrue(math.isnan(estimator([adbf])[0]))
    # Test if the register is negative.
    adbf.sketch = np.array([-1, 0])
    estimator = FirstMomentEstimator(method='uniform')
    self.assertTrue(math.isnan(estimator([adbf])[0]))

  def test_denoise_and_union(self):
    noiser = BlipNoiser(
        epsilon=math.log(3), random_state=np.random.RandomState(5))
    estimator = FirstMomentEstimator(
        method='log',
        denoiser=SurrealDenoiser(epsilon=math.log(3)))
    results = []
    truth = 1000
    for i in range(100):
      sketch_list = []
      set_ids = np.arange(truth)
      for _ in range(2):
        sketch = LogarithmicBloomFilter(length=2048, random_seed=i)
        sketch.add_ids(set_ids)
        sketch = noiser(sketch)
        sketch_list.append(sketch)
      estimate = estimator(sketch_list)
      results.append(estimate)
    self.assertAlmostEqual(truth, np.mean(results), delta=truth * 0.1)

  @parameterized.parameters(
      (UniformBloomFilter, {}, 'uniform', 2.773),
      (LogarithmicBloomFilter, {}, 'log', 4.0),
      (ExponentialBloomFilter, {'decay_rate': 1}, 'exp', 2.85),
      (GeometricBloomFilter, {'probability': 0.5}, 'geo', 2.89),
  )
  def test_estimate_cardinality_with_global_noise(
      self, bf, bf_kwargs, method, truth):
    noiser = GeometricEstimateNoiser(
        epsilon=0.5, random_state=np.random.RandomState(2))
    adbf = bf(length=4, random_seed=0, **bf_kwargs)
    adbf.add_ids([1])
    estimator = FirstMomentEstimator(method=method, noiser=noiser)
    estimate = estimator([adbf])[0]
    self.assertAlmostEqual(estimate, truth, 1, msg=method)


class FixedProbabilityBitFlipNoiserTest(absltest.TestCase):

  def test_bit_flip(self):
    rs = np.random.RandomState()
    b = BloomFilter(length=10)

    b.add(1)
    b.add(2)

    noiser = FixedProbabilityBitFlipNoiser(probability=1.0, random_state=rs)

    # Since p(flip) = 1 we are computing the inverse
    inverse = noiser(b)

    # Now do it again, which should result in a copy of the original filter b.
    inverse_of_inverse = noiser(inverse)

    result = b.sketch + inverse.sketch
    expected = np.ones(b.sketch.shape)
    self.assertTrue(
        np.array_equal(expected, result),
        'expected {} != {}'.format(expected, result))

    self.assertTrue(
        np.array_equal(b.sketch, inverse_of_inverse.sketch),
        'expected {} == {}'.format(inverse_of_inverse.sketch, b.sketch))


class BlipNoiserTest(absltest.TestCase):

  def test_probability_correct_single_hash(self):
    self.assertAlmostEqual(
        get_probability_of_flip(epsilon=math.log(3), num_hashes=1),
        0.25)

  def test_probability_correct_two_hashes(self):
    self.assertAlmostEqual(
        get_probability_of_flip(epsilon=2 * math.log(3), num_hashes=2),
        0.25)

  def test_bit_flip(self):
    b = BloomFilter(length=100000, random_seed=4)

    epsilon = math.log(3)  # Equivalent to flipping 1/4 of the bits.
    noiser = BlipNoiser(epsilon, np.random.RandomState(4))
    noised = noiser(b)
    average_bits = np.mean(noised.sketch)
    # By central limit theorem, the average_bits should be roughly following
    # N(0.25, 0.0014**2). So we set the absolute error to 0.01 which
    # is even larger than 6.5-sigma, which means that if we test it daily,
    # the expected failure will happen every 34 million years.
    self.assertAlmostEqual(average_bits, 0.25, delta=0.01)

    # Similarly, we blip 1 bits.
    b.sketch = np.bitwise_xor(b.sketch, 1)
    noised = noiser(b)
    average_bits = np.mean(noised.sketch)
    self.assertAlmostEqual(average_bits, 0.75, delta=0.01)


class SurrealDenoiserTest(absltest.TestCase):

  def test_denoiser_estimation_correct(self):
    noised_adbf = UniformBloomFilter(4, random_seed=1)
    noised_adbf.sketch[0] = 1
    denoiser = SurrealDenoiser(epsilon=math.log(3))
    denoised_adbf = denoiser([noised_adbf])[0]
    expected = np.array([1.5, -0.5, -0.5, -0.5])
    np.testing.assert_allclose(
        denoised_adbf.sketch, expected, atol=0.01)

if __name__ == '__main__':
  absltest.main()
