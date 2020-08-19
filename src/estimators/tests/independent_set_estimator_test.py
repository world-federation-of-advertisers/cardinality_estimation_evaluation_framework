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

"""Tests for independent_set_estimator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.independent_set_estimator import IndependentSetEstimator

class IndependentSetEstimatorTest(absltest.TestCase):

  def test_independent_set_estimator_empty_list(self):
    estimator = IndependentSetEstimator(LosslessEstimator(), 100)
    result = estimator([])
    self.assertEqual(result, [0])
  
  def test_independent_set_estimator_single_sketch(self):
    sketch = ExactMultiSet()
    sketch.add_ids([1, 2, 2, 3, 3, 3, 4, 5])
    estimator = IndependentSetEstimator(LosslessEstimator(), 100)
    result = estimator([sketch])
    self.assertEqual(result, [5, 2, 1])

  def test_independent_set_estimator_two_sketches_single_frequency(self):
    sketch1 = ExactMultiSet()
    sketch1.add_ids(range(50))
    sketch2 = ExactMultiSet()
    sketch2.add_ids(range(50))
    estimator = IndependentSetEstimator(LosslessEstimator(), 100)
    result = estimator([sketch1, sketch2])
    self.assertEqual(result, [75, 25])

  def test_independent_set_estimator_two_sketches_multiple_frequencies(self):
    sketch1 = ExactMultiSet()
    sketch1.add_ids(list(range(50)) + list(range(20)))
    sketch2 = ExactMultiSet()
    sketch2.add_ids(list(range(30)) + list(range(10)))
    estimator = IndependentSetEstimator(LosslessEstimator(), 100)
    result = estimator([sketch1, sketch2])
    self.assertEqual(result, [65, 34, 9, 2])

  def test_independent_set_estimator_universe_size_exceeded(self):
    sketch = ExactMultiSet()
    sketch.add_ids(range(11))
    estimator = IndependentSetEstimator(LosslessEstimator(), 10)
    with self.assertRaises(AssertionError):
        result = estimator([sketch])
    
if __name__ == '__main__':
  absltest.main()
