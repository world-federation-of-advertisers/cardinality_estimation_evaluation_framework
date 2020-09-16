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
"""Tests for wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.any_sketch import UniqueKeyFunction
from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import ExponentialSameKeyAggregator


class ExponentialSameKeyAggregatorTest(absltest.TestCase):

  def test_add(self):
    expected_unique_key_sketch = np.zeros(4, dtype=np.int32)
    expected_frequency_count_sketch = np.zeros(4, dtype=np.int32)
    ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=10, random_seed=0)
    ska.add(0)

    expected_unique_key_sketch[0] = 1
    np.testing.assert_equal(
        ska.unique_key_sketch.sketch, expected_unique_key_sketch)
    expected_frequency_count_sketch[0] = 1
    np.testing.assert_equal(
        ska.frequency_count_sketch.sketch, expected_frequency_count_sketch)

  def test_add_ids(self):
    expected_unique_key_sketch = np.zeros(4, dtype=np.int32)
    expected_frequency_count_sketch = np.zeros(4, dtype=np.int32)
    ska = ExponentialSameKeyAggregator(
        length=4, decay_rate=10, random_seed=0)
    ska.add_ids([0, 1])

    expected_unique_key_sketch[0] = UniqueKeyFunction.FLAG_COLLIDED_REGISTER
    np.testing.assert_equal(
        ska.unique_key_sketch.sketch, expected_unique_key_sketch)
    expected_frequency_count_sketch[0] = 2
    np.testing.assert_equal(
        ska.frequency_count_sketch.sketch, expected_frequency_count_sketch)


if __name__ == '__main__':
  absltest.main()
