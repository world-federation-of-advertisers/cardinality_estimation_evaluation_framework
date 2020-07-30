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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.common.analysis import relative_error
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


TEST_UNIVERSE_SIZE = 200
TEST_NUM_SETS = 3
TEST_SET_SIZE = 50
TEST_SET_SIZE_LIST = [TEST_SET_SIZE] * TEST_NUM_SETS


def fake_choice_fast(n, m, random_state=None):
  _ = random_state
  if isinstance(n, int):
    return np.arange(m)
  return n[:m]


class FakeRandomState:

  def poisson(self, lam, size):
    return lam * np.ones(size, dtype=int)

  def shuffle(self, l):
    l.reverse()


class SetGeneratorTest(parameterized.TestCase):

  @parameterized.parameters(
      (set_generator.IndependentSetGenerator, {}),
      (set_generator.ExponentialBowSetGenerator,
       {'user_activity_association': 'independent'}),
      (set_generator.ExponentialBowSetGenerator,
       {'user_activity_association': 'identical'}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'original', 'correlated_sets': 'all', 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'reversed', 'correlated_sets': 'all', 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'random', 'correlated_sets': 'all', 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'original', 'correlated_sets': 'one', 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'reversed', 'correlated_sets': 'one', 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'random', 'correlated_sets': 'one', 'shared_prop': 0.2}),
      (set_generator.HomogeneousMultiSetGenerator,
       {'freq_rate_list': np.ones_like(TEST_SET_SIZE_LIST), 'freq_cap': 2})
  )
  def test_set_generator_factory_with_num_and_size_corresponding_to_list(
      self, set_generator_class, kwargs):
    """Test generator_factory_with_num_and_size for partial set_generators.

    These set_generators take set_size_list as an argument. We test whether
    generator_factory_with_num_and_size gives the same result as directly
    specifying set_size_list in set_generator.

    Args:
      set_generator_class: class name of different set_generators.
      kwargs: kwargs expect universe_size, set_size_list and random_state, in
        each set_generator. In this test, universe_size and set_sizes are
        given by the global variables TEST_UNIVERSE_SIZE and TEST_SET_SIZE_LIST.
    """
    factory = set_generator_class.get_generator_factory_with_num_and_size(
        universe_size=TEST_UNIVERSE_SIZE, num_sets=TEST_NUM_SETS,
        set_size=TEST_SET_SIZE, **kwargs)
    gen_from_factory = factory(np.random.RandomState(1))
    gen_from_class = set_generator_class(
        universe_size=TEST_UNIVERSE_SIZE, set_sizes=TEST_SET_SIZE_LIST,
        **kwargs, random_state=np.random.RandomState(1))
    set_list_gen_from_factory = []
    for ids in gen_from_factory:
      set_list_gen_from_factory.append(list(ids))
    set_list_gen_from_class = []
    for ids in gen_from_class:
      set_list_gen_from_class.append(list(ids))
    self.assertSameElements(set_list_gen_from_factory, set_list_gen_from_class)

  @parameterized.parameters(
      (set_generator.FullyOverlapSetGenerator,
       {'num_sets': 10, 'set_size': 5}),
      (set_generator.SubSetGenerator,
       {'order': 'original', 'num_large_sets': 2, 'num_small_sets': 3,
        'large_set_size': 5, 'small_set_size': 2}),
      (set_generator.SubSetGenerator,
       {'order': 'reversed', 'num_large_sets': 2, 'num_small_sets': 3,
        'large_set_size': 5, 'small_set_size': 2}),
      (set_generator.SubSetGenerator,
       {'order': 'random', 'num_large_sets': 2, 'num_small_sets': 3,
        'large_set_size': 5, 'small_set_size': 2}),
  )
  def test_set_generator_factory_with_num_and_size_just_testing_random_state(
      self, set_generator_class, kwargs):
    """Test generator_factory_with_num_and_size for other set_generators.

    Until the previous test, this test is for the set_generators that
    do not take set_size_list as an argument. So here we just test whether
    the random state is correctly processed in the factory.

    Args:
      set_generator_class: class name of different set_generators.
      kwargs: kwargs expect universe_size and random_state in each
        set_generator. In this test, universe_size is given by the global
        variable TEST_UNIVERSE_SIZE.
    """
    factory = set_generator_class.get_generator_factory_with_num_and_size(
        universe_size=TEST_UNIVERSE_SIZE, **kwargs)
    gen_from_factory = factory(np.random.RandomState(1))
    gen_from_class = set_generator_class(
        universe_size=TEST_UNIVERSE_SIZE, **kwargs,
        random_state=np.random.RandomState(1))
    set_list_gen_from_factory = []
    for ids in gen_from_factory:
      set_list_gen_from_factory.append(list(ids))
    set_list_gen_from_class = []
    for ids in gen_from_class:
      set_list_gen_from_class.append(list(ids))
    self.assertSameElements(set_list_gen_from_factory, set_list_gen_from_class)

  @parameterized.parameters(
      (set_generator.IndependentSetGenerator, {}),
      (set_generator.ExponentialBowSetGenerator,
       {'user_activity_association': 'independent'}),
      (set_generator.ExponentialBowSetGenerator,
       {'user_activity_association': 'identical'}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'original', 'correlated_sets': 'all', 'shared_prop': 0.2}),
      (set_generator.SequentiallyCorrelatedSetGenerator,
       {'order': 'original', 'correlated_sets': 'one', 'shared_prop': 0.2}),
  )
  def test_set_generator_factory_with_set_size_list(self, set_generator_class,
                                                    kwargs):
    factory = set_generator_class.get_generator_factory_with_set_size_list(
        universe_size=TEST_UNIVERSE_SIZE, set_size_list=TEST_SET_SIZE_LIST,
        **kwargs)
    gen_from_factory = factory(np.random.RandomState(1))
    gen_from_class = set_generator_class(
        universe_size=TEST_UNIVERSE_SIZE, set_sizes=TEST_SET_SIZE_LIST,
        **kwargs, random_state=np.random.RandomState(1))
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
        universe_size=10000, set_sizes=[100] * 5, random_state=rs)
    for campaign_id in ind_gen:
      self.assertLen(campaign_id, 100)

  def test_independent_generator_constructor_different_sizes(self):
    rs = np.random.RandomState(1)
    ind_gen = set_generator.IndependentSetGenerator(
        universe_size=10000, set_sizes=[10, 10, 10, 20, 20],
        random_state=rs)
    set_ids_list = []
    for set_ids in ind_gen:
      set_ids_list.append(set_ids)
    for set_ids in set_ids_list[:3]:
      self.assertLen(set_ids, 10)
    for set_ids in set_ids_list[3:]:
      self.assertLen(set_ids, 20)

  def test_independent_generator_single_universe(self):
    rs = np.random.RandomState(1)
    ind_gen = set_generator.IndependentSetGenerator(
        universe_size=1, set_sizes=[1], random_state=rs)
    for campaign_id in ind_gen:
      self.assertLen(campaign_id, 1)
      self.assertEqual(campaign_id[0], 0)

  def test_exponential_bow_generator_constructor(self):
    rs = np.random.RandomState(1)
    # Low reach case, actual set size should be close to input set size
    eb_gen = set_generator.ExponentialBowSetGenerator(
        user_activity_association='independent',
        universe_size=10000, set_sizes=[1000] * 5, random_state=rs)
    for set_ids in eb_gen:
      re = relative_error(len(set_ids), 1000)
      self.assertLess(abs(re), 0.01,
                      msg='relative error > 0.01 in the low reach case')
    # High reach case, allow actual size to be more different from input size
    eb_gen = set_generator.ExponentialBowSetGenerator(
        user_activity_association='independent',
        universe_size=10000, set_sizes=[5000] * 5, random_state=rs)
    for set_ids in eb_gen:
      re = relative_error(len(set_ids), 5000)
      self.assertLess(abs(re), 0.2,
                      msg='relative error > 0.2 in the high reach case')

  def test_exponential_bow_generator_constructor_different_sizes(self):
    rs = np.random.RandomState(1)
    # Low reach case, actual set size should be close to input set size
    low_set_size_list = [600, 800, 1000]
    eb_gen = set_generator.ExponentialBowSetGenerator(
        user_activity_association='independent',
        universe_size=10000, set_sizes=low_set_size_list, random_state=rs)
    actual_set_size = iter(low_set_size_list)
    for set_ids in eb_gen:
      re = relative_error(len(set_ids), next(actual_set_size))
      self.assertLess(abs(re), 0.01,
                      msg='relative error > 0.01 in the low reach case')
    # High reach case, allow actual size to be more different from input size
    high_set_size_list = [4000, 5000, 6000]
    eb_gen = set_generator.ExponentialBowSetGenerator(
        user_activity_association='independent',
        universe_size=10000, set_sizes=high_set_size_list, random_state=rs)
    actual_set_size = iter(high_set_size_list)
    for set_ids in eb_gen:
      re = relative_error(len(set_ids), next(actual_set_size))
      self.assertLess(abs(re), 0.2,
                      msg='relative error > 0.2 in the high reach case')

  def test_exponential_bow_generator_raise_error(self):
    rs = np.random.RandomState(1)
    # invalid user_activity_association
    with self.assertRaises(ValueError):
      _ = set_generator.ExponentialBowSetGenerator(
          user_activity_association=0.5,
          universe_size=10000, set_sizes=[1000] * 3, random_state=rs)
    # Two small set size
    with self.assertRaises(ValueError):
      _ = set_generator.ExponentialBowSetGenerator(
          user_activity_association='independent',
          universe_size=10000, set_sizes=[10] * 3, random_state=rs)

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
        order='original', correlated_sets='all', universe_size=30,
        shared_prop=0.2, set_sizes=[10] * 3, random_state=rs)
    set_ids_list = [set_ids for set_ids in sc_gen]
    previous_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      shared_ids = previous_set_ids.intersection(set_ids)
      self.assertLen(shared_ids, 2)
      previous_set_ids.update(set_ids)

  def test_sequentially_correlated_all_previous_generator_different_sizes(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='original', correlated_sets='all', universe_size=100,
        shared_prop=0.2, set_sizes=[10, 15, 20, 20], random_state=rs)
    expected_overlap_size = iter([2, 3, 4])
    set_ids_list = [set_ids for set_ids in sc_gen]
    previous_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      shared_ids = previous_set_ids.intersection(set_ids)
      self.assertLen(shared_ids, next(expected_overlap_size))
      previous_set_ids.update(set_ids)

  def test_sequentially_correlated_all_previous_generator_reversed(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='reversed', correlated_sets='all', universe_size=30,
        shared_prop=0.2, set_sizes=[10] * 3, random_state=rs)
    set_ids_list = [set_ids for set_ids in sc_gen][::-1]
    previous_set_ids = set(set_ids_list[0])
    for set_ids in set_ids_list[1:]:
      shared_ids = previous_set_ids.intersection(set_ids)
      self.assertLen(shared_ids, 2)
      previous_set_ids.update(set_ids)

  def test_sequentially_correlated_one_previous_generator_original(self):
    rs = np.random.RandomState(1)
    sc_gen = set_generator.SequentiallyCorrelatedSetGenerator(
        order='original', correlated_sets='one', universe_size=30,
        shared_prop=0.2, set_sizes=[10] * 3, random_state=rs)
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
        order='reversed', correlated_sets='one', universe_size=30,
        shared_prop=0.2, set_sizes=[10] * 3, random_state=rs)
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
          shared_prop=0.2, set_sizes=[10] * 3, random_state=rs)
    with self.assertRaises(ValueError):
      _ = set_generator.SequentiallyCorrelatedSetGenerator(
          order='random', correlated_sets='not_implemented', universe_size=30,
          shared_prop=0.2, set_sizes=[10] * 3, random_state=rs)

  @parameterized.parameters(
      (None, [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1]]),
      (3, [[0, 0, 0], [0, 0, 1, 1]]))
  def test_homogeneous_multiset_generator_freq_cap(
      self, freq_cap, expected_multiset_ids_list):
    set_sizes = [1, 2]
    freq_rate_list = [5, 1]
    rs = FakeRandomState()
    gen = set_generator.HomogeneousMultiSetGenerator(
        universe_size=4,
        set_sizes=set_sizes,
        freq_rate_list=freq_rate_list,
        freq_cap=3,
        random_state=rs)
    expected_multiset_ids_list = [[0, 0, 0], [1, 1, 0, 0]]
    with mock.patch.object(gen, 'choice', autospec=True) as mock_choice:
      mock_choice.side_effect = fake_choice_fast
      output_multiset_ids_list = [multiset_ids for multiset_ids in gen]
      self.assertCountEqual(
          expected_multiset_ids_list,
          output_multiset_ids_list,
          'Multiset doesn\'t contain the correct IDs with freq cap:'
          f'{freq_cap}.')

  def test_homogeneous_multiset_generator_raise_unequal_length_input(self):
    # Test if raise error when set_sizes and freq_rate_list do not have
    # equal length.
    with self.assertRaises(AssertionError):
      _ = set_generator.HomogeneousMultiSetGenerator(
          universe_size=4,
          set_sizes=[1, 1],
          freq_rate_list=[1],
          freq_cap=3,
          random_state=np.random.RandomState())

  def test_homogeneous_multiset_generator_raise_invalid_freq_rate(self):
    # Test if raise error when freq_rate is invalid.
    with self.assertRaises(AssertionError):
      _ = set_generator.HomogeneousMultiSetGenerator(
          universe_size=4,
          set_sizes=[1, 1],
          freq_rate_list=[-1, 1],
          freq_cap=3,
          random_state=np.random.RandomState())

  @parameterized.parameters(0, -1)
  def test_homogeneous_multiset_generator_raise_invalid_freq_cap(self,
                                                                 freq_cap):
    # Test if raise error when freq_cap is invalid.
    with self.assertRaises(AssertionError):
      _ = set_generator.HomogeneousMultiSetGenerator(
          universe_size=4,
          set_sizes=[1, 1],
          freq_rate_list=[1, 1],
          freq_cap=freq_cap,
          random_state=np.random.RandomState())

if __name__ == '__main__':
  absltest.main()
