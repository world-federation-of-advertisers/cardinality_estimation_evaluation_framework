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

"""Tests for example_estimator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import AddRandomElementsNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LessOneEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator


class ExactSetTest(absltest.TestCase):

  def test_sketch(self):
    s = ExactSet()
    s.add_ids([1, 2])
    self.assertLen(s, 2)
    self.assertIn(2, s)
    self.assertNotIn(3, s)

  def test_lossless_estimator(self):
    s = ExactSet()
    s.add_ids([1, 2])
    e = LosslessEstimator()
    self.assertEqual(e([s])[0], 2)

  def test_less_one_estimator_multiple(self):
    s1 = ExactSet()
    s1.add_ids([1, 2])
    s2 = ExactSet()
    s2.add_ids([1, 3, 4])
    e = LessOneEstimator()
    self.assertEqual(e([s1, s2])[0], 3)

  def test_noiser(self):
    s = ExactSet()
    s.add_ids([1, 2])
    n = AddRandomElementsNoiser(
        num_random_elements=3, random_state=np.random.RandomState(1))
    s_copy = n(s)
    self.assertLen(s, 2)
    self.assertLen(s_copy, 5)


if __name__ == '__main__':
  absltest.main()
