"""Tests for noisers."""

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common import noisers
from absl.testing import absltest


class FakeRandomState:
  def __init__(self, return_value):
    self.return_value = return_value

  def laplace(self, size, scale):
    return self.return_value

class NoisersTest(absltest.TestCase):

  def test_laplace_mechanism_works_with_scipy_stats_laplace(self):
    lm = noisers.LaplaceMechanism(lambda x: np.array([1., 2., 3.]), 1., 2.)
    result = lm(5.)
    self.assertLen(result, 3)

  def test_laplace_mechanism_adds_expected_noise(self):
    rs = FakeRandomState(np.array([2., 4., 6.]))
    lm = noisers.LaplaceMechanism(lambda x: np.array([1., 2., 3.]), 1., 2., rs)
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
    rs = FakeRandomState(np.array([2., 4., 6.]))
    lm = noisers.GeometricMechanism(lambda x: np.array([1., 2., 3.]), 1., 2., rs)
    result = lm(5.)
    np.testing.assert_array_equal(result, np.array([3., 6., 9.]))

  def test_geometric_mechanism_respects_random_state(self):
    lm0 = noisers.GeometricMechanism(
        lambda x: np.array([1., 2., 3.]),
        1.,
        2.,
        random_state=np.random.RandomState(seed=125))
    result0 = lm0(5.)
    lm1 = noisers.GeometricMechanism(
        lambda x: np.array([1., 2., 3.]),
        1.,
        2.,
        random_state=np.random.RandomState(seed=125))
    result1 = lm1(5.)
    self.assertAlmostEqual(list(result0), list(result1))
    result2 = lm1(5.)
    self.assertNotEqual(list(result1), list(result2))


if __name__ == '__main__':
  absltest.main()
