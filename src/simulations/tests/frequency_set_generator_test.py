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

"""Tests for wfa_cardinality_estimation_evaluation_framework.simulations.frequency_set_generator."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.common.analysis import relative_error
import wfa_cardinality_estimation_evaluation_framework.common.random
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.simulations.frequency_set_generator import HomogeneousPmfMultiSetGenerator
from wfa_cardinality_estimation_evaluation_framework.simulations.frequency_set_generator import HomogeneousMultiSetGenerator
from wfa_cardinality_estimation_evaluation_framework.simulations.frequency_set_generator import HeterogeneousMultiSetGenerator
from wfa_cardinality_estimation_evaluation_framework.simulations.frequency_set_generator import PublisherConstantFrequencySetGenerator

class FrequencySetGeneratorTest(parameterized.TestCase):

  def test_homogeneous_pmf_multiset_generator_single_set(self):
      pmfgen = HomogeneousPmfMultiSetGenerator(
          100, [2], [[1]], np.random.RandomState())
      hists = []
      for s in pmfgen:
          e = ExactMultiSet()
          e.add_ids(s)
          hists.append(LosslessEstimator()([e]))
      self.assertLen(hists, 1)
      self.assertEqual(hists[0], [2])

  def test_homogeneous_pmf_multiset_generator_multiple_sets(self):
      pmfgen = HomogeneousPmfMultiSetGenerator(
          100, [2,1,2], [[1], [0,1], [0,0,1]], np.random.RandomState())
      sets = [s for s in pmfgen]
      hists = []
      for s in pmfgen:
          e = ExactMultiSet()
          e.add_ids(s)
          hists.append(LosslessEstimator()([e]))
      self.assertLen(hists, 3)
      self.assertEqual(hists[0], [2])
      self.assertEqual(hists[1], [1,1])
      self.assertEqual(hists[2], [2,2,2])

  def test_truncated_poisson_pmf(self):
      h = HomogeneousMultiSetGenerator(10, [1], [1], np.random.RandomState())
      e = np.exp(1)
      self.assertEqual(h._truncated_poisson_pmf(1, 1), [1])
      pmf1 = h._truncated_poisson_pmf(1,4)
      self.assertLen(pmf1, 4)
      self.assertAlmostEqual(pmf1[0], 0.3678794)
      self.assertAlmostEqual(pmf1[1], 0.3678794)
      self.assertAlmostEqual(pmf1[2], 0.1839397)
      self.assertAlmostEqual(pmf1[3], 0.0803014)
      pmf2 = h._truncated_poisson_pmf(2,3)
      self.assertLen(pmf2, 3)
      self.assertAlmostEqual(pmf2[0], 0.1353353)
      self.assertAlmostEqual(pmf2[1], 0.2706706)
      self.assertAlmostEqual(pmf2[2], 0.5939942)
  
  @parameterized.parameters((100, [1, 2], (5,1)), (3, [1,], (1,)))
  def test_homogeneous_multiset_generator_freq_cap(
      self, freq_cap, set_sizes, freq_rates):
    gen = HomogeneousMultiSetGenerator(
        universe_size=4,
        set_sizes=set_sizes,
        freq_rates=freq_rates,
        freq_cap=freq_cap,
        random_state=np.random.RandomState(1))
    output_multiset_ids_list = [multiset_ids for multiset_ids in gen]
    output_multiset_sizes = [len(set(m)) for m in output_multiset_ids_list]
    self.assertEqual(output_multiset_sizes, set_sizes)

  def test_homogeneous_multiset_generator_raise_unequal_length_input(self):
    # Test if raise error when set_sizes and freq_rate_list do not have
    # equal length.
    with self.assertRaises(AssertionError):
      _ = HomogeneousMultiSetGenerator(
          universe_size=4,
          set_sizes=[1, 1],
          freq_rates=[1],
          freq_cap=3,
          random_state=np.random.RandomState())

  def test_homogeneous_multiset_generator_raise_invalid_freq_rate(self):
    # Test if raise error when freq_rate is invalid.
    with self.assertRaises(AssertionError):
      _ = HomogeneousMultiSetGenerator(
          universe_size=4,
          set_sizes=[1, 1],
          freq_rates=[-1, 1],
          freq_cap=3,
          random_state=np.random.RandomState())

  @parameterized.parameters(0, -1)
  def test_homogeneous_multiset_generator_raise_invalid_freq_cap(self,
                                                                 freq_cap):
    # Test if raise error when freq_cap is invalid.
    with self.assertRaises(AssertionError):
      _ = HomogeneousMultiSetGenerator(
          universe_size=4,
          set_sizes=[1, 1],
          freq_rates=[1, 1],
          freq_cap=freq_cap,
          random_state=np.random.RandomState())

  def test_homogeneous_multiset_generator_factory_with_num_and_size(self):
      f = HomogeneousMultiSetGenerator.get_generator_factory_with_num_and_size(
          100, 3, 5, [1, 2, 3], 10)
      gen = f(np.random.RandomState(1))
      sets = [s for s in gen]
      self.assertLen(sets, 3)

  def test_homogeneous_multiset_generator_factory_with_set_size_list(self):
      f = HomogeneousMultiSetGenerator.get_generator_factory_with_set_size_list(
          100, [1, 2, 3], [1, 2, 3], 10)
      gen = f(np.random.RandomState(1))
      sets = [s for s in gen]
      self.assertLen(sets, 3)

  def test_heterogeneous_multi_set_generator_with_frequency_cap(self):
    g = HeterogeneousMultiSetGenerator(
        1000, [100], [(1,1)], np.random.RandomState(1), freq_cap=1)
    e = ExactMultiSet()
    for ids in g:
      e.add_ids(ids)
    h = LosslessEstimator()([e])
    self.assertEqual(h, [100])

  def test_heterogeneous_multi_set_generator_test_impression_count(self):
    g = HeterogeneousMultiSetGenerator(
        1000, [10], [(1,1)], np.random.RandomState(1))
    e = ExactMultiSet()
    for ids in g:
      e.add_ids(ids)
    h = LosslessEstimator()([e])
    self.assertEqual(h[0], 10)
    self.assertGreater(len(h), 1)

  def test_heterogeneous_multiset_generator_factory_with_num_and_size(self):
      f = HeterogeneousMultiSetGenerator.get_generator_factory_with_num_and_size(
          100, 3, 5, [(1,2), (3,4), (5,6)], 10)
      gen = f(np.random.RandomState(1))
      sets = [s for s in gen]
      self.assertLen(sets, 3)

  def test_heterogeneous_multiset_generator_factory_with_set_size_list(self):
      f = HeterogeneousMultiSetGenerator.get_generator_factory_with_set_size_list(
          100, [1, 2, 3], [(1,2), (3,4), (5,6)], 10)
      gen = f(np.random.RandomState(1))
      sets = [s for s in gen]
      self.assertLen(sets, 3)

  def test_publisher_constant_frequency_set_generator(self):
      gen = PublisherConstantFrequencySetGenerator(
          100, [1, 2, 3], 3, np.random.RandomState())
      hists = []
      for s in gen:
          e = ExactMultiSet()
          e.add_ids(s)
          hists.append(LosslessEstimator()([e]))
      self.assertLen(hists, 3)
      self.assertEqual(hists[0], [1,1,1])
      self.assertEqual(hists[1], [2,2,2])
      self.assertEqual(hists[2], [3,3,3])

  def test_publisher_constant_frequency_factory_with_num_and_size(self):
      f = PublisherConstantFrequencySetGenerator.get_generator_factory_with_num_and_size(
          100, 3, 3, 3)
      gen = f(np.random.RandomState(1))
      sets = [s for s in gen]
      self.assertLen(sets, 3)

  def test_publisher_constant_frequency_factory_with_set_size_list(self):
      f = PublisherConstantFrequencySetGenerator.get_generator_factory_with_set_size_list(
          100, [1, 2, 3], 3)
      gen = f(np.random.RandomState(1))
      sets = [s for s in gen]
      self.assertLen(sets, 3)

      
if __name__ == '__main__':
  absltest.main()
