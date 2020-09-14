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

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import ExponentialBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import GeometricBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UniformBloomFilter

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filter_sketch_operators import BayesianApproximationSketchOperator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filter_sketch_operators import ExpectationApproximationSketchOperator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filter_sketch_operators import SketchOperator


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
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM)
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
    operator = BayesianApproximationSketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM)
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

  def test_difference_bayesian_approximation(self):
    operator = BayesianApproximationSketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM)
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

  def test_intersection_expectation_approximation(self):
    operator = ExpectationApproximationSketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM)
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

  def test_difference_expectation_approximation(self):
    operator = ExpectationApproximationSketchOperator(
        estimation_method=FirstMomentEstimator.METHOD_UNIFORM)
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
