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
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import invert_monotonic
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import SketchOperator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import SurrealDenoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UniformBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UnionEstimator


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
    factory = GeometricBloomFilter.get_sketch_factory(length=15, probability=0.08)

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
      (UniformBloomFilter, {}, 'any', 1)
  )
  def test_estimate_cardinality(self, bf, bf_kwargs, method, truth):
    adbf = bf(length=4, random_seed=0, **bf_kwargs)
    adbf.add_ids([1])
    estimator = FirstMomentEstimator(method=method)
    estimate = estimator([adbf])[0]
    self.assertAlmostEqual(estimate, truth, 3, msg=method)

  def test_denoise_and_union(self):
    noiser = FixedProbabilityBitFlipNoiser(
        probability=0.25, random_state=np.random.RandomState(5))
    estimator = FirstMomentEstimator(
        method='log',
        denoiser=SurrealDenoiser(probability=0.25))
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
    print(np.mean(results))
    self.assertAlmostEqual(truth, np.mean(results), delta=truth * 0.1)


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
    rs = np.random.RandomState(2)
    epsilon = math.log(3)
    noiser = BlipNoiser(epsilon, rs)
    self.assertAlmostEqual(noiser.get_probability_of_flip(num_hashes=1), 0.25)

  def test_probability_correct_two_hashes(self):
    rs = np.random.RandomState(2)
    epsilon = 2 * math.log(3)
    noiser = BlipNoiser(epsilon, rs)
    self.assertAlmostEqual(noiser.get_probability_of_flip(num_hashes=2), 0.25)

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

  def test_denoise(self):
    noised_adbf = UniformBloomFilter(4, random_seed=1)
    noised_adbf.sketch[0] = 1
    denoiser = SurrealDenoiser(probability=0.25)
    denoised_adbf = denoiser([noised_adbf])[0]
    expected = np.array([1.5, -0.5, -0.5, -0.5])
    np.testing.assert_allclose(
        denoised_adbf.sketch, expected, atol=0.01,
        err_msg='Denoiser does not work.')


class SketchOperatorTest(absltest.TestCase):

  def test_get_register_probs(self):
    adbf = UniformBloomFilter(length=4, random_seed=1)
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    np.testing.assert_array_equal(
        SketchOperator._get_register_probs(adbf), expected,
        err_msg='Uniform ADBF register probs not as expected.')
    adbf = GeometricBloomFilter(length=4, probability=0.08, random_seed=1)
    expected = np.array([0.282, 0.260, 0.239, 0.220])
    np.testing.assert_allclose(
        SketchOperator._get_register_probs(adbf), expected, atol=0.01,
        err_msg='Geometric ADBF register probs not as expected.')
    adbf = LogarithmicBloomFilter(length=4, random_seed=1)
    expected = np.array([0.494, 0.281, 0.157, 0.0685])
    np.testing.assert_allclose(
        SketchOperator._get_register_probs(adbf), expected, atol=0.01,
        err_msg='Logarithmic ADBF register probs not as expected.')
    adbf = ExponentialBloomFilter(length=4, decay_rate=1, random_seed=1)
    expected = np.array([0.329, 0.270, 0.221, 0.181])
    np.testing.assert_allclose(
        SketchOperator._get_register_probs(adbf), expected, atol=0.01,
        err_msg='Exponential ADBF register probs not as expected.')

  def test_union(self):
    operator = SketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM,
        approximation_method=SketchOperator.BAYESIAN_APPROXIMATION)
    # Test for raw sketches
    this = UniformBloomFilter(length=2, random_seed=1)
    this.sketch[0] = 1
    that = UniformBloomFilter(length=2, random_seed=1)
    that.sketch[1] = 1
    union = operator.union(this, that)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(union.sketch, expected)
    # Test for denoised sketches
    this.sketch = np.array([-0.5, 1.5])
    that.sketch = np.array([1.5, -0.5])
    union = operator.union(this, that)
    expected = np.array([1.75, 1.75])
    np.testing.assert_array_equal(union.sketch, expected)

  def test_intersection_bayesian_approximation(self):
    operator = SketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM,
        approximation_method=SketchOperator.BAYESIAN_APPROXIMATION)
    # Test for raw sketches, simple case
    this = UniformBloomFilter(length=6, random_seed=1)
    this.sketch[0] = 1
    that = UniformBloomFilter(length=6, random_seed=1)
    that.sketch[1] = 1
    intersection = operator.intersection(this, that)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(intersection.sketch, expected)
    # Test for raw sketches, complicated case
    this.sketch = np.array([1, 0, 1, 1, 0, 0])
    that.sketch = np.array([0, 1, 1, 1, 0, 0])
    intersection = operator.intersection(this, that)
    expected = np.array([0, 0, 0.742, 0.742, 0, 0])
    np.testing.assert_allclose(intersection.sketch, expected, atol=0.01)
    # Test for denoised sketches
    this.sketch = np.array([1.5, -0.5, 1.5, 1.5, -0.5, -0.5])
    that.sketch = np.array([-0.5, 1.5, 1.5, 1.5, -0.5, -0.5])
    intersection = operator.intersection(this, that)
    expected = [-0.734, -0.734, 2.201, 2.201, 0.245, 0.245]
    np.testing.assert_allclose(intersection.sketch, expected, atol=0.01)

  def test_intersection_expectation_approximation(self):
    operator = SketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM,
        approximation_method=SketchOperator.EXPECTATION_APPROXIMATION)
    # Test for raw sketches, simple case
    this = UniformBloomFilter(length=6, random_seed=1)
    this.sketch[0] = 1
    that = UniformBloomFilter(length=6, random_seed=1)
    that.sketch[1] = 1
    intersection = operator.intersection(this, that)
    expected = np.array([0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(intersection.sketch, expected)
    # Test for raw sketches, complicated case
    this.sketch = np.array([1, 0, 1, 1, 0, 0])
    that.sketch = np.array([0, 1, 1, 1, 0, 0])
    intersection = operator.intersection(this, that)
    expected = np.array([0, 0, 0.698, 0.698, 0, 0])
    np.testing.assert_allclose(intersection.sketch, expected, atol=0.01)
    # Test for denoised sketches
    this.sketch = np.array([1.5, -0.5, 1.5, 1.5, -0.5, -0.5])
    that.sketch = np.array([-0.5, 1.5, 1.5, 1.5, -0.5, -0.5])
    intersection = operator.intersection(this, that)
    expected = [-0.614, -0.614, 1.843, 1.843, 0.205, 0.205]
    np.testing.assert_allclose(intersection.sketch, expected, atol=0.01)

  def test_difference_bayesian_approximation(self):
    operator = SketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM,
        approximation_method=SketchOperator.BAYESIAN_APPROXIMATION)
    # Test for raw sketches, simple case
    this = UniformBloomFilter(length=6, random_seed=1)
    this.sketch[0] = 1
    that = UniformBloomFilter(length=6, random_seed=1)
    that.sketch[1] = 1
    intersection = operator.difference(this, that)
    expected = np.array([1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(intersection.sketch, expected)
    # Test for raw sketches, complicated case
    this.sketch = np.array([1, 0, 1, 1, 0, 0])
    that.sketch = np.array([0, 1, 1, 1, 0, 0])
    difference = operator.difference(this, that)
    expected = np.array([1, 0, 0.524, 0.524, 0, 0])
    np.testing.assert_allclose(difference.sketch, expected, atol=0.01)
    # Test for denoised sketches
    this.sketch = np.array([1.5, -0.5, 1.5, 1.5, -0.5, -0.5])
    that.sketch = np.array([-0.5, 1.5, 1.5, 1.5, -0.5, -0.5])
    difference = operator.difference(this, that)
    expected = [2.369, 0.369, -1.106, -1.106, -0.790, -0.790]
    np.testing.assert_allclose(difference.sketch, expected, atol=0.01)

  def test_difference_expectation_approximation(self):
    operator = SketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM,
        approximation_method=SketchOperator.EXPECTATION_APPROXIMATION)
    # Test for raw sketches, simple case
    this = UniformBloomFilter(length=6, random_seed=1)
    this.sketch[0] = 1
    that = UniformBloomFilter(length=6, random_seed=1)
    that.sketch[1] = 1
    intersection = operator.difference(this, that)
    expected = np.array([1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(intersection.sketch, expected)
    # Test for raw sketches, complicated case
    this.sketch = np.array([1, 0, 1, 1, 0, 0])
    that.sketch = np.array([0, 1, 1, 1, 0, 0])
    difference = operator.difference(this, that)
    expected = np.array([1, 0, 0.441, 0.441, 0, 0])
    np.testing.assert_allclose(difference.sketch, expected, atol=0.01)
    # Test for denoised sketches
    this.sketch = np.array([1.5, -0.5, 1.5, 1.5, -0.5, -0.5])
    that.sketch = np.array([-0.5, 1.5, 1.5, 1.5, -0.5, -0.5])
    difference = operator.difference(this, that)
    expected = [2.25, 0.25, -0.75, -0.75, -0.75, -0.75]
    np.testing.assert_allclose(difference.sketch, expected, atol=0.01)

if __name__ == '__main__':
  absltest.main()
