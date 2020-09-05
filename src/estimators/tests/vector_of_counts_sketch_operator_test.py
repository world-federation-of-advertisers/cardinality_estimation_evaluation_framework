"""Tests for google3.experimental.users.huangxichen.om.voc.src.estimators.tests.vector_of_counts_sketch_operator."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import SequentialEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts_sketch_operator import SketchOperator


class SketchOperatorTest(parameterized.TestCase):

  def setUp(self):
    super(SketchOperatorTest, self).setUp()

    def _create_sketches(this_stats, that_stats):
      sketches = []
      for stats in [this_stats, that_stats]:
        sketch = None
        if stats is not None:
          sketch = VectorOfCounts(num_buckets=2, random_seed=1)
          sketch.stats = np.array(stats)
        sketches.append(sketch)
      return sketches

    self.create_sketches = _create_sketches

    def _assert_sketches_equal(result, expected):
      if expected is None:
        self.assertIsNone(result, 'The result should be None.')
      else:
        self.assertIsInstance(result, VectorOfCounts)
        expected = np.array(expected)
        np.testing.assert_array_equal(
            result.stats, expected,
            f'Expect: {expected}, get: {result.stats}.')

    self.assert_sketch_equal = _assert_sketches_equal

  @parameterized.parameters(
      ([1, 0], [0, 1], [1, 1]),
      (None, [0, 1], [0, 1]),
      ([1, 0], None, [1, 0]),
      (None, None, None),
  )
  def test_union_works(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = SketchOperator(estimator=SequentialEstimator())
    union_sketch = operator.union(*sketches)
    self.assert_sketch_equal(union_sketch, expected)

  @parameterized.parameters(
      ([1, 0], [0, 1], [-0.25, -0.25]),
      (None, [0, 1], None),
      ([1, 0], None, None),
      (None, None, None),
  )
  def test_intersection_works(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = SketchOperator(estimator=SequentialEstimator())
    intersection_sketch = operator.intersection(*sketches)
    self.assert_sketch_equal(intersection_sketch, expected)

  @parameterized.parameters(
      ([2, 1], [1, 1], [1, 0]),
      (None, [0, 1], None),
      ([1, 0], None, [1, 0]),
      (None, None, None),
  )
  def test_difference_works(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = SketchOperator(estimator=SequentialEstimator())
    difference_sketch = operator.difference(*sketches)
    self.assert_sketch_equal(difference_sketch, expected)

if __name__ == '__main__':
  absltest.main()
