"""Tests for noisers."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common import noisers


class FakeLaplaceRandomState:

  def __init__(self, return_value):
    self.return_value = return_value

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
    self.return_value = return_value

  def normal(self, size, scale):
    return self.return_value


class NoisersTest(absltest.TestCase):

  def test_laplace_mechanism_works_with_scipy_stats_laplace(self):
    lm = noisers.LaplaceMechanism(lambda x: np.array([1., 2., 3.]), 1., 2.)
    result = lm(5.)
    self.assertLen(result, 3)

  def test_laplace_mechanism_adds_expected_noise(self):
    rs = FakeLaplaceRandomState(np.array([2., 4., 6.]))
    lm = noisers.LaplaceMechanism(lambda x: np.array([1., 2., 3.]), 1., 2.,
                                  random_state=rs)
    result = lm(5.)
    np.testing.assert_array_equal(result, np.array([3., 6., 9.]))

  def test_laplace_mechanism_respects_random_state(self):
    lm0 = noisers.LaplaceMechanism(
        lambda x: np.array([1., 2., 3.]),
        1.,
        2.,
        random_state=np.random.RandomState(seed=123))
    result0 = lm0(5.)
    lm1 = noisers.LaplaceMechanism(
        lambda x: np.array([1., 2., 3.]),
        1.,
        2.,
        random_state=np.random.RandomState(seed=123))
    result1 = lm1(5.)
    self.assertEqual(list(result0), list(result1))
    result2 = lm1(5.)
    self.assertNotEqual(list(result1), list(result2))

  def test_geometric_mechanism_works_with_scipy_stats_geom(self):
    lm = noisers.GeometricMechanism(lambda x: np.array([1., 2., 3.]), 1., 2.)
    result = lm(5.)
    self.assertLen(result, 3)

  def test_geometric_mechanism_adds_expected_noise(self):
    rs = FakeGeometricRandomState(np.array([[3., 6., 13.], [1., 2., 7.]]))
    lm = noisers.GeometricMechanism(lambda x: np.array([1., 2., 3.]), 1., 2.,
                                    random_state=rs)
    result = lm(5.)
    np.testing.assert_array_equal(result, np.array([3., 6., 9.]))

  def test_geometric_mechanism_respects_random_state(self):
    lm0 = noisers.GeometricMechanism(
        lambda x: np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
        1.,
        2.,
        random_state=np.random.RandomState(seed=125))
    result0 = lm0(5.)
    lm1 = noisers.GeometricMechanism(
        lambda x: np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
        1.,
        2.,
        random_state=np.random.RandomState(seed=125))
    result1 = lm1(5.)
    self.assertAlmostEqual(list(result0), list(result1))
    result2 = lm1(5.)
    self.assertNotEqual(list(result1), list(result2))

  def test_gaussian_mechanism_works_with_scipy_stats_normal(self):
    lm = noisers.GaussianMechanism(lambda x: np.array([1., 2., 3.]), 1., 2., .1)
    result = lm(5.)
    self.assertLen(result, 3)

  def test_gaussian_mechanism_adds_expected_noise(self):
    rs = FakeGaussianRandomState(np.array([2., 4., 6.]))
    lm = noisers.GaussianMechanism(
        lambda x: np.array([1., 2., 3.]), 1., 2., .1, random_state=rs)
    result = lm(5.)
    np.testing.assert_array_equal(result, np.array([3., 6., 9.]))

  def test_gaussian_mechanism_respects_random_state(self):
    lm0 = noisers.GaussianMechanism(
        lambda x: np.array([1., 2., 3.]),
        1.,
        2.,
        .1,
        random_state=np.random.RandomState(seed=123))
    result0 = lm0(5.)
    lm1 = noisers.GaussianMechanism(
        lambda x: np.array([1., 2., 3.]),
        1.,
        2.,
        .1,
        random_state=np.random.RandomState(seed=123))
    result1 = lm1(5.)
    self.assertEqual(list(result0), list(result1))
    result2 = lm1(5.)
    self.assertNotEqual(list(result1), list(result2))


if __name__ == '__main__':
  absltest.main()
