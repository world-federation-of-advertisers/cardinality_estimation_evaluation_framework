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
"""Tests for wfa_cardinality_estimation_evaluation_framework.estimators.liquid_legions."""

from absl import logging
from absl.testing import absltest

import farmhash
import numpy

from wfa_cardinality_estimation_evaluation_framework.estimators import liquid_legions


class LiquidTest(absltest.TestCase):

  def make_chain(self, num_sketches, a, m, set_halfsize):
    result = []
    for i in range(num_sketches):
      sketch = liquid_legions.LiquidLegions(
          a, m, random_seed=42)
      for j in range(set_halfsize):
        item_1 = farmhash.hash32withseed(f'i{i}_{j}', 567)
        item_2 = farmhash.hash32withseed(f'i{i + 1}_{j}', 567)
        sketch.add_id(item_1)
        sketch.add_id(item_2)
      result.append(sketch)
    return result

  def test_pure_sketch_building(self):
    l = liquid_legions.LiquidLegions(10, 100000, random_seed=42)

    for i in range(1000000):
      if i % 10000 == 0:
        self.assertAlmostEqual(float(i), float(l.get_cardinality()),
                               delta=(i + 1) * 0.05)
      l.add_id(i)

  def test_noisy_union(self):
    dp_p = 0.25
    noiser = liquid_legions.Noiser(dp_p)
    num_sketches = 4
    halfsize = 100000
    chain = self.make_chain(num_sketches, 3, 100000, halfsize)
    noised_chain = list(map(noiser, chain))
    estimator = liquid_legions.Estimator(dp_p)
    estimation = estimator(noised_chain)
    true_size = (num_sketches + 1) * halfsize
    logging.info('Estimate: %.3f, Truth: %.3f, RelError: %.3f.',
                 estimation, true_size, estimation / true_size - 1)
    self.assertAlmostEqual(estimation, true_size, delta=true_size * 0.1)

  def test_venn_estimator_pure_single(self):
    s = liquid_legions.LiquidLegions(5, 100000, random_seed=42)
    s.add_ids(range(2000))
    e = liquid_legions.VennEstimator([s])
    self.assertAlmostEqual(e.estimate_from_all()[1], 2000, delta=50)
    logging.info('Venn: %s', e.estimate_from_all())

  def test_venn_estimator_pure(self):
    chain = self.make_chain(2, 5, 100000, 50000)
    e = liquid_legions.VennEstimator(chain)
    self.assertAlmostEqual(e.estimate_from_all()[1], 50000,
                           delta=2500)
    self.assertAlmostEqual(e.estimate_from_all()[2], 50000,
                           delta=2500)
    self.assertAlmostEqual(e.estimate_from_all()[3], 50000,
                           delta=2500)
    logging.info('Venn: %s', e.estimate_from_all())

  def test_venn_estimator_noised(self):
    chain = self.make_chain(2, 5, 100000, 50000)
    dp_p = 0.25
    noiser = liquid_legions.Noiser(dp_p)
    noised_chain = list(map(noiser, chain))
    e = liquid_legions.VennEstimator(noised_chain)
    self.assertAlmostEqual(e.estimate_from_all()[1], 50000,
                           delta=6000)
    self.assertAlmostEqual(e.estimate_from_all()[2], 50000,
                           delta=6000)
    self.assertAlmostEqual(e.estimate_from_all()[3], 50000,
                           delta=6000)
    logging.info('Venn: %s', e.estimate_from_all())

  def test_venn_priors_single_tiny(self):
    chain = self.make_chain(1, 50.0, 100, 10000)
    sampler = liquid_legions.Sampler(chain)
    priors = sampler.get_all_venn_priors()
    logging.info('Priors: %s', priors)
    self.assertAlmostEqual(priors[0][0], 0.0)
    self.assertAlmostEqual(priors[0][1], 1.0)
    self.assertAlmostEqual(priors[99][0], 1.0)
    self.assertAlmostEqual(priors[99][1], 0.0)

  def test_venn_priors_two_tiny(self):
    s1 = liquid_legions.LiquidLegions(20.0, 10, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(10)))
    s2.add_ids(list(range(10000)))

    e = liquid_legions.VennEstimator([s1, s2])
    logging.info('Venn: %s', e.estimate_from_all())
    sampler = liquid_legions.Sampler([s1, s2])
    priors = sampler.get_all_venn_priors()
    logging.info('Priors: %s', priors)
    logging.info('Row sums: %s', priors.sum(axis=1))
    self.assertAlmostEqual(priors[0][3], 1.0)
    self.assertAlmostEqual(priors[4][2], 0.95, delta=0.1)
    self.assertAlmostEqual(priors[9][0], 0.95, delta=0.1)

  def test_venn_priors_two(self):
    s1 = liquid_legions.LiquidLegions(20.0, 10000, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(1000)))
    s2.add_ids(list(range(20000)))

    e = liquid_legions.VennEstimator([s1, s2])
    logging.info('Venn: %s', e.estimate_from_all())
    sampler = liquid_legions.Sampler([s1, s2])
    priors = sampler.get_all_venn_priors()
    logging.info('Priors: %s', priors)
    counts = priors.sum(axis=0)
    logging.info('Counts: %s', counts)
    s1_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[1] + counts[3])
    s2_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[2] + counts[3])
    logging.info('Cardinalities: %.3f, %.3f', s1_cardinality, s2_cardinality)
    self.assertAlmostEqual(s1_cardinality, 1000, delta=100)
    self.assertAlmostEqual(s2_cardinality, 20000, delta=2000)

  def test_distribution_given_observation(self):
    s1 = liquid_legions.LiquidLegions(20.0, 10, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(10)))
    s2.add_ids(list(range(10000)))

    s1, s2 = map(liquid_legions.Noiser(0.01), [s1, s2])
    sampler = liquid_legions.Sampler([s1, s2])
    logging.info('Transition matrix: %s', sampler.transition_matrix)
    logging.info('Distribution: %s', sampler.distribution_given_observation(1))
    self.assertAlmostEqual(sampler.transition_matrix[0, 0], 0.98, delta=0.001)
    self.assertAlmostEqual(sampler.transition_matrix[0, 1], 0.0099,
                           delta=0.0001)

  def test_posteriors_tiny_pure(self):
    s1 = liquid_legions.LiquidLegions(20.0, 10, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(10)))
    s2.add_ids(list(range(10000)))

    e = liquid_legions.VennEstimator([s1, s2])
    logging.info('Venn: %s', e.estimate_from_all())
    sampler = liquid_legions.Sampler([s1, s2])
    posteriors = sampler.get_all_posteriors()
    logging.info('Posteriors: %s', posteriors)
    logging.info('Row sums: %s', posteriors.sum(axis=1))
    self.assertAlmostEqual(posteriors[0, 3], 1.0)
    self.assertAlmostEqual(posteriors[3, 2], 1.0)
    for x in posteriors.sum(axis=1):
      self.assertAlmostEqual(x, 1.0)

  def test_posteriors_pure(self):
    s1 = liquid_legions.LiquidLegions(2.0, 100000, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(30000)))
    s2.add_ids(list(range(20000, 40000)))

    dp_p = 0.0
    noiser = liquid_legions.Noiser(dp_p)
    s1, s2 = list(map(noiser, [s1, s2]))

    e = liquid_legions.VennEstimator([s1, s2])
    logging.info('Venn: %s', e.estimate_from_all())
    sampler = liquid_legions.Sampler([s1, s2])
    posteriors = sampler.get_all_posteriors()

    logging.info('Row sums: %s', posteriors.sum(axis=1))

    counts = posteriors.sum(axis=0)
    logging.info('Bit count expectations: %s', counts)
    s1_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[1] + counts[3])
    logging.info('s1 cardinality: %.3f',
                 s1_cardinality)
    s2_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[2] + counts[3])
    logging.info('s2 cardinality: %.3f',
                 s2_cardinality)
    union_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[1] + counts[2] + counts[3])
    logging.info('s1 | s2 cardinality: %.3f',
                 union_cardinality)
    self.assertAlmostEqual(s1_cardinality, 30000, delta=1500)
    self.assertAlmostEqual(s2_cardinality, 20000, delta=1000)
    self.assertAlmostEqual(union_cardinality, 40000, delta=2000)

  def test_posteriors_noisy(self):
    s1 = liquid_legions.LiquidLegions(2.0, 100000, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(30000)))
    s2.add_ids(list(range(20000, 40000)))

    s1 = liquid_legions.Noiser(0.3)(s1)
    s2 = liquid_legions.Noiser(0.2)(s2)

    e = liquid_legions.VennEstimator([s1, s2])
    logging.info('Venn: %s', e.estimate_from_all())
    sampler = liquid_legions.Sampler([s1, s2])
    posteriors = sampler.get_all_posteriors()
    logging.info('Row sums: %s', posteriors.sum(axis=1))

    counts = posteriors.sum(axis=0)
    logging.info('Bit count expectations: %s', counts)
    s1_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[1] + counts[3])
    logging.info('s1 cardinality: %.3f',
                 s1_cardinality)
    s2_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[2] + counts[3])
    logging.info('s2 cardinality: %.3f',
                 s2_cardinality)
    union_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[1] + counts[2] + counts[3])
    logging.info('s1 | s2 cardinality: %.3f',
                 union_cardinality)
    self.assertAlmostEqual(s1_cardinality, 30000, delta=1500)
    self.assertAlmostEqual(s2_cardinality, 20000, delta=1000)
    self.assertAlmostEqual(union_cardinality, 40000, delta=2000)

  def test_sample_matrix_noisy(self):
    s1 = liquid_legions.LiquidLegions(2.0, 100000, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(30000)))
    s2.add_ids(list(range(20000, 40000)))

    dp_p = 0.25
    noiser = liquid_legions.Noiser(dp_p)
    s1, s2 = list(map(noiser, [s1, s2]))

    e = liquid_legions.VennEstimator([s1, s2])
    logging.info('Venn: %s', e.estimate_from_all())
    sampler = liquid_legions.Sampler([s1, s2])
    posteriors = sampler.get_all_posteriors()
    logging.info('Row sums: %s', posteriors.sum(axis=1))

    sample = sampler.sample_matrix()
    logging.info('Sample shape: %s', sample.shape)
    counts = sample.sum(axis=0)
    logging.info('Bit count expectations: %s', counts)
    s1_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[0])
    s2_cardinality = sampler.sketch.get_cardinality_for_legionaries_count(
        counts[1])
    logging.info('s1 cardinality: %s', s1_cardinality)
    logging.info('s2 cardinality: %s', s2_cardinality)
    self.assertAlmostEqual(s1_cardinality, 30000, delta=1500)
    self.assertAlmostEqual(s2_cardinality, 20000, delta=1000)

  def test_sample_noisy(self):
    s1 = liquid_legions.LiquidLegions(2.0, 100000, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(30000)))
    s2.add_ids(list(range(20000, 40000)))

    noised_s1 = liquid_legions.Noiser(0.25)(s1)
    noised_s2 = liquid_legions.Noiser(0.1)(s2)

    sampler = liquid_legions.Sampler([noised_s1, noised_s2])

    m = sampler.sample_matrix()
    logging.info('Sampled matrix marginal: %s', m.sum(axis=0))

    sampled_s1, sampled_s2 = sampler.sample()

    logging.info('s1 legionaries: %d', sampled_s1.legionaries_count())
    logging.info('s1 cardinality: %.3f', sampled_s1.get_cardinality())
    logging.info('s2 cardinality: %.3f', sampled_s2.get_cardinality())
    union = liquid_legions.LiquidLegions.merge_of([s1, s2])
    logging.info('s1|s2 cardinality: %.3f',
                 union.get_cardinality())
    self.assertAlmostEqual(sampled_s1.get_cardinality(), 30000, delta=1500)
    self.assertAlmostEqual(sampled_s2.get_cardinality(), 20000, delta=1000)
    self.assertAlmostEqual(union.get_cardinality(), 40000, delta=2000)

  def test_sample_noisy_large(self):
    s1 = liquid_legions.LiquidLegions(2.0, 100000, random_seed=42)
    s2 = s1.get_compatible_sketch()

    s1.add_ids(list(range(300000)))
    s2.add_ids(list(range(200000, 400000)))

    noised_s1 = liquid_legions.Noiser(0.25)(s1)
    noised_s2 = liquid_legions.Noiser(0.1)(s2)

    sampler = liquid_legions.Sampler([noised_s1, noised_s2])

    m = sampler.sample_matrix()
    logging.info('Sampled matrix: %s', m.sum(axis=0))

    sampled_s1, sampled_s2 = sampler.sample()

    logging.info('s1 legionaries: %d', sampled_s1.legionaries_count())
    logging.info('s1 cardinality: %.3f', sampled_s1.get_cardinality())
    logging.info('s2 cardinality: %.3f', sampled_s2.get_cardinality())
    union = liquid_legions.LiquidLegions.merge_of([s1, s2])
    logging.info('s1|s2 cardinality: %.3f',
                 union.get_cardinality())
    self.assertAlmostEqual(sampled_s1.get_cardinality(), 300000, delta=15000)
    self.assertAlmostEqual(sampled_s2.get_cardinality(), 200000, delta=10000)
    self.assertAlmostEqual(union.get_cardinality(), 400000, delta=20000)

  def test_manual_sequential_merge_small_overlap(self):
    noiser = liquid_legions.Noiser(0.25)
    universe_set = range(50000)

    s = liquid_legions.LiquidLegions(10.0, 50000, random_seed=42)

    true_set = set()
    for i in range(10):
      a = s.get_compatible_sketch()
      new_set = numpy.random.choice(universe_set, size=1000)
      a.add_ids(new_set)
      true_set = true_set | set(new_set)
      noised_a = noiser(a)
      sampler = liquid_legions.Sampler([s, noised_a])
      _, sampled_a = sampler.sample()
      venn_estimator = liquid_legions.VennEstimator([s, noised_a])
      logging.info('Venn: %s', venn_estimator())

      s.merge_in(sampled_a)
      logging.info('Step %d, cardinality %.3f, true cardinality: %d',
                   i, s.get_cardinality(), len(true_set))
    self.assertAlmostEqual(len(true_set), s.get_cardinality(),
                           delta=len(true_set) * 0.2)

  def test_manual_sequential_merge_large_overlap(self):
    noiser = liquid_legions.Noiser(0.05)  # Small noise!

    s = liquid_legions.LiquidLegions(10.0, 50000, random_seed=42)

    true_set = set()
    for i in range(10):
      a = s.get_compatible_sketch()
      new_set = range(i * 1000, i * 1000 + 10000)
      a.add_ids(new_set)
      true_set = true_set | set(new_set)
      noised_a = noiser(a)
      sampler = liquid_legions.Sampler([s, noised_a])
      _, sampled_a = sampler.sample()
      venn_estimator = liquid_legions.VennEstimator([s, noised_a])
      logging.info('Venn: %s', venn_estimator())

      s.merge_in(sampled_a)
      logging.info('Step %d, cardinality %.3f, true cardinality: %d',
                   i, s.get_cardinality(), len(true_set))
    self.assertAlmostEqual(len(true_set), s.get_cardinality(),
                           delta=len(true_set) * 1.0)  # Large bound!

  def test_manual_sequential_merge_large_overlap_pure(self):
    noiser = liquid_legions.Noiser(0.0)  # No noise.

    s = liquid_legions.LiquidLegions(10.0, 50000, random_seed=42)

    true_set = set()
    for i in range(10):
      a = s.get_compatible_sketch()
      new_set = range(i * 1000, i * 1000 + 10000)
      a.add_ids(new_set)
      true_set = true_set | set(new_set)
      noised_a = noiser(a)
      sampler = liquid_legions.Sampler([s, noised_a])
      _, sampled_a = sampler.sample()
      venn_estimator = liquid_legions.VennEstimator([s, noised_a])
      logging.info('Venn: %s', venn_estimator())

      s.merge_in(sampled_a)
      logging.info('Step %d, cardinality %.3f, true cardinality: %d',
                   i, s.get_cardinality(), len(true_set))
    self.assertAlmostEqual(len(true_set), s.get_cardinality(),
                           delta=len(true_set) * 0.1)

  def test_sequential_estimator(self):
    noiser = liquid_legions.Noiser(0.25)
    estimator = liquid_legions.SequentialEstimator()
    true_set = set()
    universe_set = list(range(1000000))
    sketches = []
    for i in range(10):
      new_set = numpy.random.choice(universe_set, size=10000)
      sketch = liquid_legions.LiquidLegions(10, 10000, random_seed=42)
      sketch.add_ids(new_set)
      true_set = true_set | set(new_set)
      noised_sketch = noiser(sketch)
      sketches.append(noised_sketch)
      logging.info('Step %d, cardinality %.3f, true_cardinality %d.',
                   i, estimator(sketches), len(true_set))
    self.assertAlmostEqual(len(true_set),
                           estimator(sketches),
                           delta=len(true_set) * 0.25)

  def test_set_difference_pure(self):
    s1 = liquid_legions.LiquidLegions(3.0, 100000, random_seed=42)
    s2 = liquid_legions.LiquidLegions(3.0, 100000, random_seed=42)
    s1.add_ids(list(range(0, 60000)))
    s2.add_ids(list(range(40000, 150000)))
    # Expected diff is range(0, 40000), i.e. size 40k.
    sampler = liquid_legions.Sampler([s1, s2])
    estimated_diff_sketch = sampler.sample_diff()
    diff_sketch = s1.get_compatible_sketch()
    diff_sketch.add_ids(range(0, 40000))
    logging.info('Diff cardinality: %.3f',
                 estimated_diff_sketch.get_cardinality())
    self.assertAlmostEqual(estimated_diff_sketch.get_cardinality(),
                           40000, delta=2000)
    match_count = {}
    for i in range(100000):
      estimated_bit = estimated_diff_sketch.sketch.get(i, 0) > 0
      true_bit = diff_sketch.sketch.get(i, 0) > 0
      match_count[true_bit, estimated_bit] = match_count.get(
          (true_bit, estimated_bit), 0) + 1
    logging.info('Match / mismatch counts: %s', sorted(match_count.items()))

  def test_set_difference_noised(self):
    noiser = liquid_legions.Noiser(0.25)
    s1 = liquid_legions.LiquidLegions(3.0, 100000, random_seed=42)
    s2 = liquid_legions.LiquidLegions(3.0, 100000, random_seed=42)
    s1.add_ids(list(range(0, 60000)))
    s2.add_ids(list(range(40000, 150000)))
    # Expected diff is range(0, 40000), i.e. size 40k.
    sampler = liquid_legions.Sampler([noiser(s1), noiser(s2)])
    estimated_diff_sketch = sampler.sample_diff()
    diff_sketch = s1.get_compatible_sketch()
    diff_sketch.add_ids(range(0, 40000))
    logging.info('Diff cardinality: %.3f',
                 estimated_diff_sketch.get_cardinality())
    self.assertAlmostEqual(estimated_diff_sketch.get_cardinality(),
                           40000, delta=4000)
    match_count = {}
    for i in range(100000):
      estimated_bit = estimated_diff_sketch.sketch.get(i, 0) > 0
      true_bit = diff_sketch.sketch.get(i, 0) > 0
      match_count[true_bit, estimated_bit] = match_count.get(
          (true_bit, estimated_bit), 0) + 1
    logging.info('Match / mismatch counts: %s', sorted(match_count.items()))

if __name__ == '__main__':
  absltest.main()
