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

"""Tests for wfa_cardinality_estimation_evaluation_framework.simulations.set_generator."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


class SetGeneratorTest(parameterized.TestCase):

  @parameterized.parameters(
      (set_generator.IndependentSetGenerator,
       {'universe_size': 10, 'num_sets': 10, 'set_size': 5}),
      # Exponential (Dirac) Bow does not support too small set size
      (set_generator.ExponentialBowSetGenerator,
       {'user_activity_association': 'independent', 'universe_size': 200,
        'num_sets': 5, 'set_size': 100}),
      (set_generator.ExponentialBowSetGenerator,
       {'user_activity_association': 'identical', 'universe_size': 200,
        'num_sets': 5, 'set_size': 100}),
      (set_generator.FullyOverlapSetGenerator,
       {'universe_size': 10, 'num_sets': 10, 'set_size': 5}),
      (set_generator.SubSetGenerator,
       {'order': 'original', 'universe_size': 10, 'num_large_sets': 2,
        'num_small_sets': 3, 'large_set_size': 5, 'small_set_size': 2}),
      (set_generator.SubSetGenerator,
       {'order': 'reversed', 'universe_size': 10, 'num_large_sets': 2,
        'num_small_sets': 3, 'large_set_size': 5, 'small_set_size': 2}),
      (set_generator.SubSetGenerator,
       {'order': 'random', 'universe_size': 10, 'num_large_sets': 2,
        'num_small_sets': 3, 'large_set_size': 5, 'small_set_size': 2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'original', 'correlated_sets': 'all', 'universe_size': 30,
        'num_sets': 3, 'set_size': 5, 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'reversed', 'correlated_sets': 'all', 'universe_size': 30,
        'num_sets': 3, 'set_size': 5, 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'random', 'correlated_sets': 'all', 'universe_size': 30,
        'num_sets': 3, 'set_size': 5, 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'original', 'correlated_sets': 'one', 'universe_size': 30,
        'num_sets': 3, 'set_size': 5, 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'reversed', 'correlated_sets': 'one', 'universe_size': 30,
        'num_sets': 3, 'set_size': 5, 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'random', 'correlated_sets': 'one', 'universe_size': 30,
        'num_sets': 3, 'set_size': 5, 'shared_prop': 0.2}),
  )
  def test_set_generator_factory(self, set_generator_class, kwargs):
    factory = set_generator_class.get_generator_factory(**kwargs)
    gen_from_factory = factory(np.random.RandomState(1))
    gen_from_class = set_generator_class(
        random_state=np.random.RandomState(1),
        **kwargs)
    set_list_gen_from_factory = []
    for ids in gen_from_factory:
      set_list_gen_from_factory.append(list(ids))
    set_list_gen_from_class = []
    for ids in gen_from_class:
      set_list_gen_from_class.append(list(ids))
    self.assertSameElements(set_list_gen_from_factory, set_list_gen_from_class)

  def test_independent_generator_constructor(self):
    rs = np.random.RandomState(1)
    ind_gen = set_generator.IndependentSetGenerator(
        universe_size=10000, num_sets=5, set_size=100, random_state=rs)
    for campaign_id in ind_gen:
      self.assertLen(campaign_id, 100)

  def test_independent_generator_single_universe(self):
    rs = np.random.RandomState(1)
    ind_gen = set_generator.IndependentSetGenerator(
        universe_size=1, num_sets=1, set_size=1, random_state=rs)
    for campaign_id in ind_gen:
      self.assertLen(campaign_id, 1)
      self.assertEqual(campaign_id[0], 0)

  def test_exponential_bow_generator_constructor(self):
    rs = np.random.RandomState(1)
    # Low reach case, actual set size should be close to input set size
    eb_gen = set_generator.ExponentialBowSetGenerator(
        user_activity_association='independent',
        universe_size=10000, num_sets=5, set_size=1000, random_state=rs)
    for set_ids in eb_gen:
      relative_error = (len(set_ids) - 1000) / 1000
      self.assertLess(abs(relative_error), 0.01,
                      msg='relative error > 0.01 in the low reach case')
    # High reach case, allow actual size to be more different from input size
    eb_gen = set_generator.ExponentialBowSetGenerator(
        user_activity_association='independent',
        universe_size=10000, num_sets=5, set_size=5000, random_state=rs)
    for set_ids in eb_gen:
      relative_error = (len(set_ids) - 5000) / 5000
      self.assertLess(abs(relative_error), 0.2,
                      msg='relative error > 0.2 in the high reach case')

  def test_exponential_bow_generator_raise_error(self):
    rs = np.random.RandomState(1)
    with self.assertRaises(ValueError):
      _ = set_generator.ExponentialBowSetGenerator(
          user_activity_association=0.5,
          universe_size=10000, num_sets=5, set_size=1000, random_state=rs)

  def test_fully_overlap_generator_constructor(self):
    rs = np.random.RandomState(1)
    fo_gen = set_generator.FullyOverlapSetGenerator(
        universe_size=10000, num_sets=5, set_size=100, random_state=rs)
    for set_ids in fo_gen:
      self.assertLen(set_ids, 100)

  def test_fully_overlap_generator_single_universe(self):
    rs = np.random.RandomState(1)
    fo_gen = set_generator.FullyOverlapSetGenerator(
        universe_size=1, num_sets=1, set_size=1, random_state=rs)
    for set_ids in fo_gen:
      self.assertLen(set_ids, 1)
      self.assertEqual(set_ids[0], 0)

  def test_fully_overlap_generator_same_ids(self):
    rs = np.random.RandomState(1)
    fo_gen = set_generator.FullyOverlapSetGenerator(
        universe_size=10, num_sets=10, set_size=5, random_state=rs)
    set_ids_list = []
    for set_ids in fo_gen:
      set_ids_list.append(set_ids)
    for set_ids in set_ids_list[1:]:
      self.assertSameElements(set_ids_list[0], set_ids)
      self.assertLen(set_ids, 5)

  def test_subset_generator_constructor_original_order(self):
    rs = np.random.RandomState(1)
    ss_gen = set_generator.SubSetGenerator(
        order='original', universe_size=10000, num_large_sets=2,
        num_small_sets=8, large_set_size=100, small_set_size=50,
        random_state=rs)
    set_ids_list = []
    for set_ids in ss_gen:
      set_ids_list.append(set_ids)
    for set_ids in set_ids_list[:2]:
      self.assertLen(set_ids, 100)
    for set_ids in set_ids_list[2:]:
      self.assertLen(set_ids, 50)

  def test_subset_generator_constructor_reversed_order(self):
    rs = np.random.RandomState(1)
    ss_gen = set_generator.SubSetGenerator(
        order='reversed', universe_size=10000, num_large_sets=2,
        num_small_sets=8, large_set_size=100, small_set_size=50,
        random_state=rs)
    set_ids_list = []
    for set_ids in ss_gen:
      set_ids_list.append(set_ids)
    for set_ids in set_ids_list[:8]:
      self.assertLen(set_ids, 50)
    for set_ids in set_ids_list[8:]:
      self.assertLen(set_ids, 100)

  def test_subset_generator_constructor_random_order(self):
    rs = np.random.RandomState(1)
    ss_gen = set_generator.SubSetGenerator(
        order='random', universe_size=10000, num_large_sets=2,
        num_small_sets=8, large_set_size=100, small_set_size=50,
        random_state=rs)
    set_ids_list = []
    for set_ids in ss_gen:
      set_ids_list.append(set_ids)
    actual_num_large_sets = 0
    actual_num_small_sets = 0
    for set_ids in set_ids_list:
      if len(set_ids) == 100:
        actual_num_large_sets += 1
      if len(set_ids) == 50:
        actual_num_small_sets += 1
    self.assertEqual(actual_num_large_sets, 2,
                     msg='Number of large sets is not correct.')
    self.assertEqual(actual_num_small_sets, 8,
                     msg='Number of small sets is not correct.')

  def test_subset_generator_single_universe(self):
    rs = np.random.RandomState(1)
    ss_gen = set_generator.SubSetGenerator(
        order='original', universe_size=1, num_large_sets=1,
        num_small_sets=1, large_set_size=1, small_set_size=1,
        random_state=rs)
    for set_ids in ss_gen:
      self.assertLen(set_ids, 1)
      self.assertEqual(set_ids[0], 0)

  def test_subset_generator_same_ids(self):
    rs = np.random.RandomState(1)
    ss_gen = set_generator.SubSetGenerator(
        order='original', universe_size=10, num_large_sets=3,
        num_small_sets=7, large_set_size=5, small_set_size=3,
        random_state=rs)
    set_ids_list = []
    for set_ids in ss_gen:
      set_ids_list.append(set_ids)
    for set_ids in set_ids_list[1:3]:
      self.assertSameElements(set_ids_list[0], set_ids)
      self.assertLen(set_ids, 5)
    for set_ids in set_ids_list[4:]:
      self.assertSameElements(set_ids_list[3], set_ids)
      self.assertLen(set_ids, 3)

  def test_subset_generator_generator_raise_error(self):
    rs = np.random.RandomState(1)
    with self.assertRaises(ValueError):
      _ = set_generator.SubSetGenerator(
          order='not_implemented', universe_size=10, num_large_sets=3,
          num_small_sets=7, large_set_size=5, small_set_size=3,
          random_state=rs)

  def test_sequentially_correlated_all_previous_generator_original(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='original', correlated_sets='all', universe_size=30, num_sets=3,
        set_size=10, shared_prop=0.2, random_state=rs)
    set_ids_list = [set_ids for set_ids in sc_gen]
    previous_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      shared_ids = previous_set_ids.intersection(set_ids)
      self.assertLen(shared_ids, 2)
      previous_set_ids.update(set_ids)

  def test_sequentially_correlated_all_previous_generator_reversed(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='reversed', correlated_sets='all', universe_size=30, num_sets=3,
        set_size=10, shared_prop=0.2, random_state=rs)
    set_ids_list = [set_ids for set_ids in sc_gen][::-1]
    previous_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      shared_ids = previous_set_ids.intersection(set_ids)
      self.assertLen(shared_ids, 2)
      previous_set_ids.update(set_ids)

  def test_sequentially_correlated_one_previous_generator_original(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='original', correlated_sets='one', universe_size=30, num_sets=3,
        set_size=10, shared_prop=0.2, random_state=rs)
    set_ids_list = [set_ids for set_ids in sc_gen]
    previous_set_ids = set(set_ids_list[0])
    union_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      self.assertLen(previous_set_ids.intersection(set_ids), 2)
      self.assertLen(union_set_ids.intersection(set_ids), 2)
      previous_set_ids = set(set_ids)
      union_set_ids.update(set_ids)

  def test_sequentially_correlated_one_previous_generator_reversed(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='reversed', correlated_sets='one', universe_size=30, num_sets=3,
        set_size=10, shared_prop=0.2, random_state=rs)
    set_ids_list = [set_ids for set_ids in sc_gen][::-1]
    previous_set_ids = set(set_ids_list[0])
    union_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      self.assertLen(previous_set_ids.intersection(set_ids), 2)
      self.assertLen(union_set_ids.intersection(set_ids), 2)
      previous_set_ids = set(set_ids)
      union_set_ids.update(set_ids)

  def test_sequentially_correlated_all_previous_generator_raise_error(self):
    rs = np.random.RandomState(1)
    with self.assertRaises(ValueError):
      _ = set_generator.SequentiallyCorrelatedSetGenerator(
          order='not_implemented', correlated_sets='all', universe_size=30,
          num_sets=3, set_size=10, shared_prop=0.2, random_state=rs)
    with self.assertRaises(ValueError):
      _ = set_generator.SequentiallyCorrelatedSetGenerator(
          order='random', correlated_sets='not_implemented', universe_size=30,
          num_sets=3, set_size=10, shared_prop=0.2, random_state=rs)


if __name__ == '__main__':
  absltest.main()
