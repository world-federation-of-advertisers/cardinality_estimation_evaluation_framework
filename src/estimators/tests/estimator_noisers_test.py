"""Tests for estimator_noisers."""

import numpy as np
import scipy.stats

from wfa_cardinality_estimation_evaluation_framework.estimators import estimator_noisers
from absl.testing import absltest


class FakeLaplaceRandomState:

  def __init__(self, return_value):
    self.return_value = np.array(return_value)

  def laplace(self, size, scale):
    return self.return_value


class FakeGeometricRandomState:

  def __init__(self, return_values):
    self.return_values = return_values
    self.calls_count = 0

  def geometric(self, size, p):
    self.calls_count += 1
    return self.return_values[self.calls_count - 1]


class FakeGaussianRandomState:

  def __init__(self, return_value):
    self.return_value = np.array(return_value)

  def normal(self, size, scale):
    return self.return_value


class FakeDiscreteGaussianRandomState:

  def __init__(self, return_values_geometric, return_values_binomial):
    self.return_values_geometric = return_values_geometric
    self.return_values_binomial = return_values_binomial
    self.calls_count_geometric = 0
    self.calls_count_binomial = 0

  def geometric(self, p):
    self.calls_count_geometric += 1
    return self.return_values_geometric[self.calls_count_geometric - 1]

  def binomial(self, n, p):
    self.calls_count_binomial += 1
    return self.return_values_binomial[self.calls_count_binomial - 1]


class EstimatorNoisersTest(absltest.TestCase):

  def test_laplace_estimate_noiser_accepts_scalar_argument(self):
    le = estimator_noisers.LaplaceEstimateNoiser(
        1.0, random_state=FakeLaplaceRandomState([0.5]))
    result = le(10.)
    self.assertEqual(result, 10.5)

  def test_laplace_estimate_noiser_accepts_array_argument(self):
    le = estimator_noisers.LaplaceEstimateNoiser(
        1.0, random_state=FakeLaplaceRandomState([0.5, -0.5]))
    result = le(np.array([10., 20.]))
    np.testing.assert_array_equal(result, [10.5, 19.5])

  def test_geometric_estimate_noiser_accepts_scalar_argument(self):
    le = estimator_noisers.GeometricEstimateNoiser(
        1.0, random_state=FakeGeometricRandomState([[4.], [3.]]))
    result = le(10.)
    self.assertEqual(result, 11)

  def test_geometric_estimate_noiser_accepts_array_argument(self):
    le = estimator_noisers.GeometricEstimateNoiser(
        1.0, random_state=FakeGeometricRandomState([[3., -2.], [2., -1.]]))
    result = le(np.array([10., 20.]))
    np.testing.assert_array_equal(result, [11, 19])

  def test_gaussian_estimate_noiser_accepts_scalar_argument(self):
    le = estimator_noisers.GaussianEstimateNoiser(
        1., .1, random_state=FakeGaussianRandomState([0.5]))
    result = le(10.)
    self.assertEqual(result, 10.5)

  def test_gaussian_estimate_noiser_accepts_array_argument(self):
    le = estimator_noisers.GaussianEstimateNoiser(
        1., .1, random_state=FakeGaussianRandomState([0.5, -0.5]))
    result = le(np.array([10., 20.]))
    np.testing.assert_array_equal(result, [10.5, 19.5])

  def test_discrete_gaussian_estimate_noiser_accepts_scalar_argument(self):
    le = estimator_noisers.DiscreteGaussianEstimateNoiser(
        1., .1, random_state=FakeDiscreteGaussianRandomState([1.5, 1.], [1]))
    result = le(10.)
    self.assertEqual(result, 10.5)

  def test_discrete_gaussian_estimate_noiser_accepts_array_argument(self):
    le = estimator_noisers.DiscreteGaussianEstimateNoiser(
        1., .1, random_state=FakeDiscreteGaussianRandomState(
          [1.5, 1., -10., -30., 15., 13.7], [1, 0, 1]))
    result = le(np.array([10., 20.]))
    np.testing.assert_array_equal(result, [10.5, 21.3])


if __name__ == '__main__':
  absltest.main()
