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
"""Tests for wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts_sketch_operator."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts_sketch_operator import StratifiedSketchOperator


class StratifiedSketchOperatorTest(parameterized.TestCase):

  def setUp(self):
    super(StratifiedSketchOperatorTest, self).setUp()

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
  def test_union(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = StratifiedSketchOperator()
    union_sketch = operator.union(*sketches)
    self.assert_sketch_equal(union_sketch, expected)

  @parameterized.parameters(
      ([1, 0], [0, 1], [-0.25, -0.25]),
      (None, [0, 1], None),
      ([1, 0], None, None),
      (None, None, None),
      ([0, 0], [1, 1], [0, 0]),
      ([1, 1], [0, 0], [0, 0]),
      ([1, 0], [0, -1], [0.25, 0.25]),
  )
  def test_intersection_no_clip(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = StratifiedSketchOperator()
    intersection_sketch = operator.intersection(*sketches)
    self.assert_sketch_equal(intersection_sketch, expected)

  @parameterized.parameters(
      ([1, 0], [0, 1], [0, 0]),  # Clip the intersection.
      (None, [0, 1], None),
      ([1, 0], None, None),
      (None, None, None),
      ([0, 0], [1, 1], [0, 0]),
      ([1, 1], [0, 0], [0, 0]),
      ([1, 0], [0, -1], [0, 0]),  # Clip the intersection.
      ([1e5, 1], [1e5, 0], [1e5, 0]),  # Has full overlap.
      ([1e5, 1], [1e4, 0], [1e4, 0]),  # No clip applied.
  )
  def test_intersection_with_clip(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = StratifiedSketchOperator(clip=True, epsilon=np.log(3),
                                        clip_threshold=1e-5)
    intersection_sketch = operator.intersection(*sketches)
    self.assert_sketch_equal(intersection_sketch, expected)

  @parameterized.parameters(
      ([2, 1], [1, 1], [1, 0]),
      (None, [0, 1], None),
      ([1, 0], None, [1, 0]),
      (None, None, None),
  )
  def test_difference(self, this_stats, that_stats, expected):
    sketches = self.create_sketches(this_stats, that_stats)
    operator = StratifiedSketchOperator()
    difference_sketch = operator.difference(*sketches)
    self.assert_sketch_equal(difference_sketch, expected)

if __name__ == '__main__':
  absltest.main()
