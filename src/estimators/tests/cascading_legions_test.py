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
"""Tests for wfa_cardinality_estimation_evaluation_framework.legions."""

import random

from absl import logging
from absl.testing import absltest
import numpy

from wfa_cardinality_estimation_evaluation_framework.estimators import cascading_legions


class LegionsTest(absltest.TestCase):

  # Set to 1 for quick sanity checks. Shoud be checked in as 5.
  NUM_ATTEMPTS = 1

  def setUp(self):
    super().setUp()
    # Avoid flakiness.
    numpy.random.seed(42)
    random.seed(42)

  def test_pure_sketch(self):
    for attempt in range(self.NUM_ATTEMPTS):
      logging.info('Attempt %s.', attempt)
      s = cascading_legions.CascadingLegions(10, 10000, random_seed=5)
      for i in range(100000):
        s.add_id(f'attempt-{attempt}-id-{i}')
      self.assertAlmostEqual(s.get_cardinality(), 10 ** 5, delta=5 * 10 ** 3)

  def test_pure_union(self):
    for attempt in range(self.NUM_ATTEMPTS):
      logging.info('Attempt %s.', attempt)
      s1 = cascading_legions.CascadingLegions(10, 10000, random_seed=5)
      s2 = cascading_legions.CascadingLegions(10, 10000, random_seed=5)
      for i in range(100000):
        s1.add_id(f'a-{i}')
        s1.add_id(f'b-{i}')
        s2.add_id(f'a-{i}')
        s2.add_id(f'c-{i}')
      s = cascading_legions.CascadingLegions(10, 10000, random_seed=5)
      s.merge_in(s1)
      s.merge_in(s2)
      true_cardinality = 3 * 10 ** 5
      self.assertAlmostEqual(
          s.get_cardinality(),
          true_cardinality, delta=true_cardinality * 0.05)

  def test_noisy_union(self):
    noiser = cascading_legions.Noiser(0.25)
    estimator = cascading_legions.Estimator(0.25)
    num_sketches = 2
    for attempt in range(self.NUM_ATTEMPTS):
      logging.info('Attempt %s.', attempt)
      sketch_list = [
          cascading_legions.CascadingLegions(20, 10000, random_seed=5)
          for _ in range(num_sketches)
      ]
      for sketch_index, s in enumerate(sketch_list):
        logging.info('Filling sketch %s.', sketch_index)
        for i in range(100000):
          s.add_id(f'common-{i}')
          s.add_id(f'{sketch_index}-{i}')
      noised_sketch_list = list(map(noiser, sketch_list))
      true_cardinality = (num_sketches + 1) * 10 ** 5
      estimate = estimator(noised_sketch_list)[0]
      logging.info(
          'True cardinality: %s, estimate: %s', true_cardinality, estimate)
      self.assertAlmostEqual(
          estimate,
          true_cardinality, delta=0.1 * true_cardinality)


if __name__ == '__main__':
  absltest.main()
