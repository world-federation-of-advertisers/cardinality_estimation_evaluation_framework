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
"""Tests for wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs."""
from unittest import mock
from absl.testing import absltest

import numpy as np

from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


class EvaluationConfigTest(absltest.TestCase):

  def test_generate_configs_scenario_3b_set_sizes_correct(self):
    conf_list = evaluation_configs._generate_configs_scenario_3b(
        universe_size=200,
        num_sets=3,
        small_set_size=50,
        large_set_size=100,
        user_activity_assciation=(set_generator
                                  .USER_ACTIVITY_ASSOCIATION_IDENTICAL)
    )

    result = {}
    for conf in conf_list:
      gen = conf.set_generator_factory(np.random.RandomState(1))
      result[conf.name] = [len(set_ids) for set_ids in gen]

    expected = {
        'exponential_bow-user_activity_association:identical-'
        'universe_size:200-small_set:50-large_set:100-set_type:all_small': [
            48, 48, 48],
        'exponential_bow-user_activity_association:identical-'
        'universe_size:200-small_set:50-large_set:100-set_type:all_large': [
            84, 84, 84],
        'exponential_bow-user_activity_association:identical-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:1st_small_then_large': [48, 84, 84],
        'exponential_bow-user_activity_association:identical-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:1st_half_small_2nd_half_large': [48, 84, 84],
        'exponential_bow-user_activity_association:identical-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:small_then_last_large': [48, 48, 84],
        'exponential_bow-user_activity_association:identical-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:gradually_smaller': [84, 67, 55]
    }

    self.assertEqual(result, expected)

  def test_generate_configs_scenario_4a_set_sizes_correct(self):
    conf_list = evaluation_configs._generate_configs_scenario_4a(
        universe_size=10,
        num_sets=3,
        small_set_size=2,
        large_set_size=8
    )

    result = {}
    for conf in conf_list:
      gen = conf.set_generator_factory(np.random.RandomState(1))
      result[conf.name] = [len(set_ids) for set_ids in gen]

    expected = {
        'fully_overlapped-universe_size:10-num_sets:3-set_sizes:2': [2, 2, 2],
        'fully_overlapped-universe_size:10-num_sets:3-set_sizes:8': [8, 8, 8]
    }

    self.assertEqual(result, expected)

  def test_generate_configs_scenario_4b_set_sizes_correct(self):
    conf_list = evaluation_configs._generate_configs_scenario_4b(
        universe_size=10,
        num_sets=3,
        small_set_size=2,
        large_set_size=8,
        order=set_generator.ORDER_REVERSED
    )

    result = {}
    for conf in conf_list:
      gen = conf.set_generator_factory(np.random.RandomState(1))
      result[conf.name] = [len(set_ids) for set_ids in gen]

    expected = {
        'subset-universe_size:10-order:reversed-num_large_sets:1-'
        'num_small_sets:2-large_set_size:8-small_set_size:2': [2, 2, 8],
        'subset-universe_size:10-order:reversed-num_large_sets:2-'
        'num_small_sets:1-large_set_size:8-small_set_size:2': [2, 8, 8]
    }

    self.assertEqual(result, expected)

  def test_generate_configs_scenario_5(self):
    conf_list = evaluation_configs._generate_configs_scenario_5(
        universe_size=100,
        num_sets=3,
        small_set_size=2,
        large_set_size=8,
        order=set_generator.ORDER_ORIGINAL,
        shared_prop_list=[0.1]
    )

    result = {}
    for conf in conf_list:
      gen = conf.set_generator_factory(np.random.RandomState(1))
      result[conf.name] = [len(set_ids) for set_ids in gen]

    expected = {
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_half_large_2nd_half_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_half_small_2nd_half_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_large_then_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_small_then_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:all_large_except_middle_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:all_small_except_middle_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:large_then_last_small-'
        'large_set_size:8-small_set_size:2': [8, 8, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:repeated_small_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:small_then_last_large-'
        'large_set_size:8-small_set_size:2': [2, 2, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_half_large_2nd_half_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_half_small_2nd_half_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_large_then_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_small_then_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:all_large_except_middle_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 8],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:all_small_except_middle_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:large_then_last_small-'
        'large_set_size:8-small_set_size:2': [8, 8, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:repeated_small_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-universe_size:100-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:small_then_last_large-'
        'large_set_size:8-small_set_size:2': [2, 2, 8]
    }

    self.assertEqual(result, expected)

  @mock.patch('wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs._generate_configs_scenario_3b')
  @mock.patch('wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs._generate_configs_scenario_4a')
  @mock.patch('wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs._generate_configs_scenario_4b')
  @mock.patch('wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs._generate_configs_scenario_5')
  def test_complete_test_with_selected_parameters_all_scenario_used(
      self,
      scenario_config_3b,
      scenario_config_4a,
      scenario_config_4b,
      scenario_config_5):
    """Test all the scenarios are concluded in the complete test."""
    _ = evaluation_configs._complete_test_with_selected_parameters()
    self.assertTrue(scenario_config_3b.called, 'Scenario 3b not included.')
    self.assertTrue(scenario_config_4a.called, 'Scenario 4a not included.')
    self.assertTrue(scenario_config_4b.called, 'Scenario 4b not included.')
    self.assertTrue(scenario_config_5.called, 'Scenario 5 not included.')


if __name__ == '__main__':
  absltest.main()
