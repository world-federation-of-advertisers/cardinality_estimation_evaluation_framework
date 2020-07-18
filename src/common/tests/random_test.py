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
"""Tests for wfa_cardinality_estimation_evaluation_framework.common.random."""
from absl.testing import absltest
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.common import random

class PlottingTest(absltest.TestCase):
  def test_choice_fast_same_random_state_same_output(self):
    rs1 = np.random.RandomState(1)
    rs2 = np.random.RandomState(1)
    a = random.choice_fast(10000, 5000, rs1)
    b = random.choice_fast(10000, 5000, rs2)
    self.assertSameElements(a, b)

  def test_choice_fast_len_is_m(self):
    for i in range(1000):
      a = random.choice_fast(10000, i)
      self.assertLen(a, i)

  def test_choice_fast_choose_elements_from_list(self):
    for i in range(50, 500):
      # Get a random list of numbers from 0 to 5000 size i
      elements = np.random.randint(0, 5000, i)
      # Choose up to i elements from that list
      chosen = random.choice_fast(elements, np.random.randint(1, i))
      # Make sure chosen elements are actually from our original elements.
      for element in chosen:
        self.assertTrue(element in elements)

  def test_choice_fast_is_unique(self):
    for i in range(50, 500):
      chosen = random.choice_fast(500, i)
      no_repeats = set(chosen)
      self.assertTrue(len(chosen) == len(no_repeats))

if __name__ == '__main__':
  absltest.main()
