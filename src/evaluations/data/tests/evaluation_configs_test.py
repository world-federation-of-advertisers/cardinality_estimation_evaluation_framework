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
import math
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.evaluations import configs
from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import _stress_test_cardinality_global_dp
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import _complete_test_with_selected_parameters
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import GAUSSIAN_NOISE
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import GLOBAL_DP_STR
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import LOCAL_DP_STR
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import NO_GLOBAL_DP_STR
from wfa_cardinality_estimation_evaluation_framework.evaluations.data.evaluation_configs import NO_LOCAL_DP_STR
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


class EvaluationConfigTest(parameterized.TestCase):

  EVALUATION_CONFIGS_MODULE = (
      'wfa_cardinality_estimation_evaluation_framework.evaluations.data.'
      + 'evaluation_configs.')

  def test_smoke_test(self):
    configs = evaluation_configs._smoke_test(4)
    self.assertEqual(configs.name, 'smoke_test')
    self.assertEqual(configs.num_runs, 4)
    self.assertLen(configs.scenario_config_list, 5)

  def test_frequency_smoke_test(self):
    configs = evaluation_configs._frequency_smoke_test(4)
    self.assertEqual(configs.name, 'frequency_smoke_test')
    self.assertEqual(configs.num_runs, 4)
    self.assertLen(configs.scenario_config_list, 3)

  def test_complete_frequency_test_with_selected_parameters(self):
    configs = evaluation_configs._complete_frequency_test_with_selected_parameters(4)
    self.assertEqual(configs.name, 'complete_frequency_test_with_selected_parameters')
    self.assertEqual(configs.num_runs, 4)

  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_freq_configs_scenario_1')
  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_freq_configs_scenario_2')
  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_freq_configs_scenario_3')
  def test_complete_frequency_test_with_selected_parameters_all_scenarios_used(
      self,
      scenario_config_1,
      scenario_config_2,
      scenario_config_3):
    """Test all the scenarios are concluded in the complete test."""
    _ = evaluation_configs._complete_frequency_test_with_selected_parameters(
        num_runs=1)
    self.assertTrue(scenario_config_1.called, 'Scenario 1 not included.')
    self.assertTrue(scenario_config_2.called, 'Scenario 2 not included.')
    self.assertTrue(scenario_config_3.called, 'Scenario 3 not included.')

  @parameterized.parameters(
      (2000, None, 'independent-universe_size:2000'),
      (2000, 0.2, 'remarketing-remarketing_size:400-universe_size:2000'),
  )
  def test_generate_configs_scenario_1_2_set_sizes_correct(
    self, universe_size, remarketing_rate, type_header
  ):
    conf_list = evaluation_configs._generate_configs_scenario_1_2(
        universe_size=universe_size,
        num_sets=3,
        small_set_size=50,
        large_set_size=100,
        remarketing_rate=remarketing_rate,
    )
    result = {}
    for conf in conf_list:
      gen = conf.set_generator_factory(np.random.RandomState(1))
      result[conf.name] = [len(set_ids) for set_ids in gen]

    expected = {
      f'{type_header}-small_set:50-large_set:100-'
      'set_type:1st_half_small_2nd_half_large': [50, 100, 100],
      f'{type_header}-small_set:50-large_set:100-'
      'set_type:1st_small_then_large': [50, 100, 100],
      f'{type_header}-small_set:50-large_set:100-'
      'set_type:all_large': [100, 100, 100],
      f'{type_header}-small_set:50-large_set:100-'
      'set_type:all_small': [50, 50, 50],
      f'{type_header}-small_set:50-large_set:100-'
      'set_type:small_then_last_large': [50, 50, 100],
      f'{type_header}-small_set:50-large_set:100-'
      'set_type:gradually_smaller': [100, 70, 57],
    }

    self.assertEqual(result, expected)

  @parameterized.parameters(
      (set_generator.USER_ACTIVITY_ASSOCIATION_INDEPENDENT),
      (set_generator.USER_ACTIVITY_ASSOCIATION_IDENTICAL),
  )
  def test_generate_configs_scenario_3_set_sizes_correct(self, activity):
    conf_list = evaluation_configs._generate_configs_scenario_3(
        universe_size=200,
        num_sets=3,
        small_set_size=50,
        large_set_size=100,
        user_activity_assciation=(activity)
    )

    result = {}
    for conf in conf_list:
      gen = conf.set_generator_factory(np.random.RandomState(1))
      result[conf.name] = [len(set_ids) for set_ids in gen]

    expected = {
        f'exponential_bow-user_activity_association:{activity}-'
        'universe_size:200-small_set:50-large_set:100-set_type:all_small': [
            48, 48, 48],
        f'exponential_bow-user_activity_association:{activity}-'
        'universe_size:200-small_set:50-large_set:100-set_type:all_large': [
            84, 84, 84],
        f'exponential_bow-user_activity_association:{activity}-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:1st_small_then_large': [48, 84, 84],
        f'exponential_bow-user_activity_association:{activity}-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:1st_half_small_2nd_half_large': [48, 84, 84],
        f'exponential_bow-user_activity_association:{activity}-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:small_then_last_large': [48, 48, 84],
        f'exponential_bow-user_activity_association:{activity}-'
        'universe_size:200-small_set:50-large_set:100-'
        'set_type:gradually_smaller': [84, 66, 55]
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

  @parameterized.parameters(
      (100000, None, None, "geo_bloom_filter-100000_0.000020"
      "-first_moment_geo-no_local_dp-no_global_dp"),
      (250000, None, None, "geo_bloom_filter-250000_0.000008"
      "-first_moment_geo-no_local_dp-no_global_dp"),
      (250000, math.log(3), math.log(3), "geo_bloom_filter-"
      "250000_0.000008-first_moment_geo-no_local_dp-no_global_dp"),
      (250000, math.log(3), None, "geo_bloom_filter-"
      "250000_0.000008-first_moment_geo-no_local_dp-no_global_dp"),
      (250000, None, math.log(3), "geo_bloom_filter-250000_0.000008"
      "-first_moment_geo-no_local_dp-no_global_dp"),
  )
  def test_geo_bloom_filter_first_moment_geo(self,
      length, sketch_epsilon, estimate_epsilon, expected):
    conf = evaluation_configs._geo_bloom_filter_first_moment_geo(
        length, None, None)
    self.assertEqual(conf.name, expected)

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
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_half_large_2nd_half_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_half_small_2nd_half_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_large_then_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:1st_small_then_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:all_large_except_middle_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:all_small_except_middle_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:large_then_last_small-'
        'large_set_size:8-small_set_size:2': [8, 8, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:repeated_small_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:small_then_last_large-'
        'large_set_size:8-small_set_size:2': [2, 2, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_half_large_2nd_half_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_half_small_2nd_half_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_large_then_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:1st_small_then_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:all_large_except_middle_small-'
        'large_set_size:8-small_set_size:2': [8, 2, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:all_small_except_middle_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:large_then_last_small-'
        'large_set_size:8-small_set_size:2': [8, 8, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:repeated_small_large-'
        'large_set_size:8-small_set_size:2': [2, 8, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:small_then_last_large-'
        'large_set_size:8-small_set_size:2': [2, 2, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:all_large-'
        'large_set_size:8-small_set_size:2': [8, 8, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:all_small-'
        'large_set_size:8-small_set_size:2': [2, 2, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:all-shared_prop:0.1-'
        'set_type:gradually_smaller-'
        'large_set_size:8-small_set_size:2': [8, 5, 4],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:all_large-'
        'large_set_size:8-small_set_size:2': [8, 8, 8],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:all_small-'
        'large_set_size:8-small_set_size:2': [2, 2, 2],
        'sequentially_correlated-order:original-'
        'correlated_sets:one-shared_prop:0.1-'
        'set_type:gradually_smaller-'
        'large_set_size:8-small_set_size:2': [8, 5, 4],
    }

    self.assertEqual(result, expected)

  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_configs_scenario_1_2')
  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_configs_scenario_3')
  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_configs_scenario_4a')
  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_configs_scenario_4b')
  @mock.patch(EVALUATION_CONFIGS_MODULE + '_generate_configs_scenario_5')
  def test_complete_test_with_selected_parameters_all_scenario_used(
      self,
      scenario_config_1_2,
      scenario_config_3,
      scenario_config_4a,
      scenario_config_4b,
      scenario_config_5):
    """Test all the scenarios are concluded in the complete test."""
    _ = _complete_test_with_selected_parameters(num_runs=1)
    self.assertTrue(scenario_config_1_2.called, 'Scenario 1/2 not included.')
    self.assertTrue(scenario_config_3.called, 'Scenario 3 not included.')
    self.assertTrue(scenario_config_4a.called, 'Scenario 4a not included.')
    self.assertTrue(scenario_config_4b.called, 'Scenario 4b not included.')
    self.assertTrue(scenario_config_5.called, 'Scenario 5 not included.')

  def test_complete_test_with_selected_parameters_contains_scenario_configs(
      self):
    eval_configs = _complete_test_with_selected_parameters(num_runs=1)
    for scenario_config in eval_configs.scenario_config_list:
      self.assertIsInstance(scenario_config, configs.ScenarioConfig)

  def test_cardinality_global_dp_stress_test(self):
    eval_configs = _stress_test_cardinality_global_dp(
        universe_size=None, num_runs=1)
    for scenario_config in eval_configs.scenario_config_list:
      self.assertIsInstance(scenario_config, configs.ScenarioConfig)
      gen = scenario_config.set_generator_factory(np.random.RandomState(0))
      reach = float(scenario_config.name.split('-')[1].lstrip('reach:'))
      set_ids_list = [set_ids for set_ids in gen]
      self.assertLen(set_ids_list, 1)
      self.assertLen(set_ids_list[0], reach)

  @parameterized.parameters(
      (LOCAL_DP_STR, None, NO_LOCAL_DP_STR),
      (GLOBAL_DP_STR, None, NO_GLOBAL_DP_STR),
      (LOCAL_DP_STR, math.log(3), LOCAL_DP_STR + '_1.0986'),
      (GLOBAL_DP_STR, math.log(3), GLOBAL_DP_STR + '_1.0986'),
      (LOCAL_DP_STR, '1.09861', LOCAL_DP_STR + '_1.0986'),
      (GLOBAL_DP_STR, '1.09861', GLOBAL_DP_STR + '_1.0986'),
      (LOCAL_DP_STR, 0, LOCAL_DP_STR + '_0.0000'),
      (GLOBAL_DP_STR, 0, GLOBAL_DP_STR + '_0.0000'),
  )
  def test_format_epsilon_correct(self, dp_type, epsilon, expected):
    self.assertEqual(evaluation_configs._format_epsilon(dp_type, epsilon),
                     expected)

  @parameterized.parameters(
      (LOCAL_DP_STR, None, None, None, None, NO_LOCAL_DP_STR),
      (GLOBAL_DP_STR, None, None, None, None, NO_GLOBAL_DP_STR),
      (LOCAL_DP_STR, math.log(3), None, None, None,
       LOCAL_DP_STR + '_1.099,0.0000'),
      (GLOBAL_DP_STR, math.log(3), 0.1, None, None,
       GLOBAL_DP_STR + '_1.099,0.1000'),
      (LOCAL_DP_STR, 1.09, 0.01, 3, None,
       LOCAL_DP_STR + '_1.090,0.0100-budget_split-3'),
      (GLOBAL_DP_STR, 3.055, 0.001, 5, GAUSSIAN_NOISE,
       GLOBAL_DP_STR + '_3.055,0.0010-' + GAUSSIAN_NOISE + '-budget_split-5'),
  )
  def test_format_privacy_parameters_correct(
    self, dp_type, epsilon, delta, num_queries, noise_type, expected):
    self.assertEqual(evaluation_configs._format_privacy_parameters(
      dp_type, epsilon=epsilon, delta=delta, num_queries=num_queries,
      noise_type=noise_type, epsilon_decimals=3, delta_decimals=4),
    expected)

  def test_construct_sketch_estimator_config_name_cardinality_estimator(self):
    name = evaluation_configs.construct_sketch_estimator_config_name(
        sketch_name='vector_of_counts',
        sketch_config='4096',
        estimator_name='sequential',
        sketch_epsilon=None,
        estimate_epsilon=1,
    )
    expected = (
        'vector_of_counts-4096-sequential'
        f'-{NO_LOCAL_DP_STR}'
        f'-{GLOBAL_DP_STR}_1.0000')
    self.assertEqual(name, expected)

  def test_construct_sketch_estimator_config_name_frequency_estimator(self):
    name = evaluation_configs.construct_sketch_estimator_config_name(
        sketch_name='exact_set',
        sketch_config='1000',
        estimator_name='lossless',
        max_frequency=5,
    )
    expected = (
        'exact_set-1000-lossless'
        f'-{NO_LOCAL_DP_STR}'
        f'-{NO_GLOBAL_DP_STR}'
        '-5')
    self.assertEqual(name, expected)

  @parameterized.parameters(
      ('vector-of_counts', '4096', 'sequential'),
      ('vector_of_counts', '4096-0', 'sequential'),
      ('vector_of_counts', '4096', 'pairwise-sequential')
  )
  def test_construct_sketch_estimator_config_raise_invalid_input(
      self, sketch_name, sketch_config, estimator_name):
    with self.assertRaises(AssertionError):
      evaluation_configs.construct_sketch_estimator_config_name(
          sketch_name, sketch_config, estimator_name)

  def test_independent_set_estimator_estimate_correct(self):
    """Test if the independent set estimator's reach estimate is correct."""
    sketch_estimator_config = evaluation_configs._independent_set_estimator()
    sketches = []
    reach = 1000
    for _ in range(2):
      sketch = sketch_estimator_config.sketch_factory(1)
      sketch.add_ids(list(range(reach)))
      sketches.append(sketch)
    estimated = sketch_estimator_config.estimator(sketches)

    # Calculate the true reach.
    reach_rate = reach / evaluation_configs.UNIVERSE_SIZE_VALUE
    expected = [
        (1 - (1 - reach_rate)**2),  # Reach percent of freq >= 1.
        reach_rate**2,  # Reach percent of freq >= 2.
    ]
    expected = [x * evaluation_configs.UNIVERSE_SIZE_VALUE for x in expected]
    for x, y in zip(estimated, expected):
      self.assertAlmostEqual(x, y)

  def test_exp_bloom_filter_first_moment_exp_with_budget_split(self):
    conf = evaluation_configs._exp_bloom_filter_first_moment_exp(
        length=5, estimate_epsilon=0.5, estimate_delta=0.03,
        noise_type=GAUSSIAN_NOISE, num_estimate_queries=25)
    self.assertEqual(
        conf.name,
        'exp_bloom_filter-5_10-first_moment_exp-no_local_dp-global_dp_'
        f'0.5000,0.0300000-gaussian_noise-budget_split-25')

  def test_exp_bloom_filter_first_moment_exp(self):
    conf = evaluation_configs._exp_bloom_filter_first_moment_exp(
        length=8, sketch_epsilon=1, estimate_epsilon=2,
        epsilon_decimals=2)
    self.assertEqual(
        conf.name,
        'exp_bloom_filter-8_10-first_moment_exp-local_dp_1.00'
        '-global_dp_2.00')

  def test_meta_voc_for_exp_adbf(self):
    conf = evaluation_configs._meta_voc_for_exp_adbf(
        adbf_length=16,
        adbf_decay_rate=2,
        voc_length=4,
        sketch_epsilon=1)
    self.assertEqual(
        conf.name,
        'exp_bloom_filter-16_2-meta_voc_4-local_dp_1.0000-no_global_dp',
        'Config name is not correct.')
    exp_adbf = conf.sketch_factory(1)
    self.assertLen(exp_adbf.sketch, 16)
    self.assertEqual(conf.estimator.adbf_estimator._method, 'exp')
    voc_sketch = conf.estimator.meta_sketch_factory(1)
    self.assertLen(voc_sketch.stats, 4)

  def test_meta_voc_for_bf(self):
    conf = evaluation_configs._meta_voc_for_bf(
        bf_length=16,
        voc_length=4)
    self.assertEqual(
        conf.name,
        'bloom_filter-16-meta_voc_4-no_local_dp-no_global_dp',
        'Config name is not correct.')
    bf = conf.sketch_factory(1)
    self.assertLen(bf.sketch, 16)
    bf.add(1)
    self.assertAlmostEqual(conf.estimator([bf])[0], 1.03, delta=1.03 * 0.1)
    voc_sketch = conf.estimator.meta_sketch_factory(1)
    self.assertLen(voc_sketch.stats, 4)

  def test_get_estimator_configs_return_configs(self):
    expected_sketch_estimator_configs = [conf.name for conf in (
        evaluation_configs._generate_cardinality_estimator_configs())]
    sketch_estimator_configs = evaluation_configs.get_estimator_configs(
        expected_sketch_estimator_configs, 1)
    self.assertLen(sketch_estimator_configs,
                   len(expected_sketch_estimator_configs))
    for conf in sketch_estimator_configs:
      self.assertIn(conf.name, expected_sketch_estimator_configs)

  def test_stratiefied_sketch_vector_of_counts(self):
    conf = evaluation_configs._stratiefied_sketch_vector_of_counts(
        max_frequency=3,
        clip=True,
        length=1024,
        sketch_epsilon=math.log(3))
    self.assertEqual(conf.max_frequency, 3)
    self.assertEqual(
        conf.name,
        'stratified_sketch_vector_of_counts-1024-sequential_clip'
        '-local_dp_1.0986-no_global_dp-3')

  def test_exp_same_key_aggregator(self):
    conf = evaluation_configs._exp_same_key_aggregator(
        max_frequency=3, length=1e2, global_epsilon=1)
    self.assertEqual(conf.max_frequency, 3)
    self.assertEqual(
        conf.name,
        'exp_same_key_aggregator-100_10-standardized_histogram'
        '-no_local_dp-global_dp_1.0000-3')

  def test_stratiefied_sketch_exp_adbf(self):
    conf = evaluation_configs._stratiefied_sketch_exponential_adbf(
        max_frequency=3, length=100, sketch_epsilon=1, global_epsilon=1,
        sketch_operator_type='expectation')
    self.assertEqual(conf.max_frequency, 3)
    self.assertEqual(
        conf.name,
        'stratified_sketch_exp_adbf-100_10'
        '-first_moment_estimator_exp_expectation'
        '-local_dp_1.0000-global_dp_1.0000-3')

  def test_stratiefied_sketch_exp_adbf_sketch_operator(self):
    for sketch_operator_type in (evaluation_configs.SKETCH_OPERATOR_EXPECTATION,
                                 evaluation_configs.SKETCH_OPERATOR_BAYESIAN):
      conf = evaluation_configs._stratiefied_sketch_exponential_adbf(
          max_frequency=3, length=100, sketch_epsilon=1, global_epsilon=1,
          sketch_operator_type=sketch_operator_type)
      self.assertIn(sketch_operator_type, conf.name)

    with self.assertRaises(ValueError):
      _ = evaluation_configs._stratiefied_sketch_exponential_adbf(
          max_frequency=3, length=100, sketch_epsilon=1, global_epsilon=1,
          sketch_operator_type='other_type')

  @parameterized.parameters(
      (3, 100, 1, 1, 'stratified_sketch_geo_adbf-'
      '100_0.020000-first_moment_estimator_geo_expectation-local_dp_1.0000-'
      'global_dp_1.0000-3'),
      (3, 4000, 1, None, 'stratified_sketch_geo_adbf-'
      '4000_0.000500-first_moment_estimator_geo_expectation-local_dp_1.0000-'
      'no_global_dp-3'),
  )
  def test_stratiefied_sketch_geo_adbf(
    self, frequency, length, sketch_epsilon, global_epsilon, truth):
    conf = evaluation_configs._stratiefied_sketch_geo_adbf(
        frequency, length, sketch_epsilon, global_epsilon)
    self.assertEqual(conf.max_frequency, frequency)
    self.assertEqual(
        conf.name,
        truth)


if __name__ == '__main__':
  absltest.main()
