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
"""Evaluation configurations."""
import itertools
import math

import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators import bloom_filter_sketch_operators
from wfa_cardinality_estimation_evaluation_framework.estimators import bloom_filters
from wfa_cardinality_estimation_evaluation_framework.estimators import estimator_noisers
from wfa_cardinality_estimation_evaluation_framework.estimators import exact_set
from wfa_cardinality_estimation_evaluation_framework.estimators import hyper_log_log
from wfa_cardinality_estimation_evaluation_framework.estimators import independent_set_estimator
from wfa_cardinality_estimation_evaluation_framework.estimators import liquid_legions
from wfa_cardinality_estimation_evaluation_framework.estimators import meta_estimators
from wfa_cardinality_estimation_evaluation_framework.estimators import same_key_aggregator
from wfa_cardinality_estimation_evaluation_framework.estimators import stratified_sketch
from wfa_cardinality_estimation_evaluation_framework.estimators import vector_of_counts
from wfa_cardinality_estimation_evaluation_framework.estimators import vector_of_counts_sketch_operator
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import EvaluationConfig
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import ScenarioConfig
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import SketchEstimatorConfig
from wfa_cardinality_estimation_evaluation_framework.simulations import frequency_set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator

SKETCH = 'sketch'
SKETCH_CONFIG = 'sketch_config'
SKETCH_EPSILON = 'sketch_epsilon'
ESTIMATOR = 'estimator'
ESTIMATE_EPSILON = 'estimate_epsilon'
MAX_FREQUENCY = 'max_frequency'

SKETCH_ESTIMATOR_CONFIG_NAMES_FORMAT = (
    SKETCH, SKETCH_CONFIG, ESTIMATOR, SKETCH_EPSILON, ESTIMATE_EPSILON,
    MAX_FREQUENCY)

NUM_RUNS_VALUE = 100
SMOKE_TEST_UNIVERSE_SIZE = 200_000
UNIVERSE_SIZE_VALUE = 1_000_000
NUM_SETS_VALUE = 20

# Smoke test reach percent.
SMALL_REACH_RATE_SMOKE_TEST = 0.1
LARGE_REACH_RATE_SMOKE_TEST = 0.2

# Complete reach estimator evaluation reach percent.
SMALL_REACH_RATE_VALUE = 0.01
LARGE_REACH_RATE_VALUE = 0.2

# Global DP stress test.
US_INTERNET_POPULATION = 2_000_000_000
REACH_STRESS_TEST = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

# Frequency test reach percent.
REACH_RATE_FREQ_END_TO_END_TEST = 0.1
REACH_RATE_FREQ_SMOKE_TEST = 0.1

SHARED_PROP_LIST_VALUE = (0.25, 0.5, 0.75)
REMARKETING_RATE_VALUE = 0.2

NUM_SETS_VALUE_FREQ = 10
SET_SIZE_FOR_FREQ = 20_000
FREQ_UNIVERSE_SIZE = 200_000

NO_GLOBAL_DP_STR = 'no_global_dp'
GLOBAL_DP_STR = 'global_dp'
NO_LOCAL_DP_STR = 'no_local_dp'
LOCAL_DP_STR = 'local_dp'

GEOMETRIC_NOISE = 'geometric_noise'
GAUSSIAN_NOISE = 'gaussian_noise'

# The None in the epsilon value is used to tell the sketch estimator constructor
# that we do not want to noise the sketch.
SKETCH_EPSILON_VALUES = (math.log(3), math.log(3) / 4, math.log(3) / 10, None)
# The current simulator module add noise to the estimated cardinality so as to
# mimic the global differential privacy use case. In the real world, the
# implementation could be different and more complicated.
# As such, we use a small epsilon so as to be conservative on the result.
ESTIMATE_EPSILON_VALUES = (math.log(3), None)
GLOBAL_DP_LIMIT_TEST_EPSILON_VALUES = [
    math.log(3) / x for x in [
        1, 2, 4, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
]
# When we would like to split budget over multiple queries, having delta greater
# than zero is often helpful in allowing us to use smaller amount of noise.
ESTIMATE_EPSILON_DELTA_VALUES = [
    (math.log(3), 1e-5), (math.log(3), 1e-6), (math.log(3), 1e-7), (None, None)
]
# The number of estimate queries for which the budget will be split over.
NUM_ESTIMATE_QUERIES_VALUES = [
    1, 2, 4, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000, 2_000,
    3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000, 50_000, 100_000,
    500_000, 1_000_000
]

# The number of decimal points to keep of the epsilon part of the
# SketchEstimatorConfig name.
EPSILON_DECIMALS = 4
EPSILON_DECIMALS_LIMIT_TEST = 8
# The number of decimal points to keep of the delta part of the
# SketchEstimatorConfig name.
DELTA_DECIMALS = 7

# The length of the Any Distribution Bloom Filters.
# We use the np.array with dtype so as to make sure that the lengths are all
# integers.
ADBF_LENGTH_LIST = np.array([100_000, 250_000], dtype=np.int64)
EXP_ADBF_DECAY_RATE = 10
STRATIFIED_EXP_ADBF_EPSILON_SPLIT = 0.5
SKETCH_OPERATOR_EXPECTATION = 'expectation'
SKETCH_OPERATOR_BAYESIAN = 'bayesian'
SKETCH_OPERATOR_LIST = [SKETCH_OPERATOR_EXPECTATION, SKETCH_OPERATOR_BAYESIAN]
GEO_LENGTH_PROB_PRODUCT = 2

# The length of the bloom filters.
BLOOM_FILTERS_LENGTH_LIST = np.array([5_000_000], dtype=np.int64)
VOC_LENGTH_LIST = np.array([1024, 4096], dtype=np.int64)


# Document the evaluation configurations.
def _smoke_test(num_runs=NUM_RUNS_VALUE,
                universe_size=SMOKE_TEST_UNIVERSE_SIZE):
  """Smoke test evaluation configurations.

  We set the smoke test parameters according to Appendix 3: Example
  Parameters of Scenarios of the Cardinality and Frequency Estimation
  Evaluation Framework.

  Args:
    num_runs: the number of runs per scenario.
    universe_size: the size of universe.

  Returns:
    An EvaluationConfig.
  """
  set_size = int(universe_size * LARGE_REACH_RATE_SMOKE_TEST)
  seq_corr_set_size = int(universe_size * SMALL_REACH_RATE_SMOKE_TEST)
  return EvaluationConfig(
      name='smoke_test',
      num_runs=num_runs,
      scenario_config_list=(
          ScenarioConfig(
              name='independent',
              set_generator_factory=(
                  set_generator.IndependentSetGenerator.
                  get_generator_factory_with_num_and_size(
                      universe_size=universe_size, num_sets=20,
                      set_size=set_size))),
          ScenarioConfig(
              name='remarketing',
              set_generator_factory=(
                  set_generator.IndependentSetGenerator.
                  get_generator_factory_with_num_and_size(
                      universe_size=int(universe_size * REMARKETING_RATE_VALUE),
                      num_sets=20, set_size=set_size))),
          ScenarioConfig(
              name='fully_overlapping',
              set_generator_factory=(
                  set_generator.FullyOverlapSetGenerator.
                  get_generator_factory_with_num_and_size(
                      universe_size=universe_size, num_sets=20,
                      set_size=set_size))),
          ScenarioConfig(
              name='sequentially_correlated_all',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory_with_num_and_size(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ALL,
                      num_sets=20, set_size=seq_corr_set_size,
                      shared_prop=0.5))),
          ScenarioConfig(
              name='sequentially_correlated_one',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory_with_num_and_size(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ONE,
                      num_sets=20, set_size=seq_corr_set_size,
                      shared_prop=0.5))),
          )
      )


def _frequency_smoke_test(num_runs=NUM_RUNS_VALUE,
                          universe_size=FREQ_UNIVERSE_SIZE):
  """Smoke test frequency evaluation configurations.

  Args:
    num_runs: the number of runs per scenario.
    universe_size: the size of the universe.

  Returns:
    An EvaluationConfig.
  """
  set_size = int(universe_size * REACH_RATE_FREQ_SMOKE_TEST)
  return EvaluationConfig(
      name='frequency_smoke_test',
      num_runs=num_runs,
      scenario_config_list=(
          ScenarioConfig(
              name='homogeneous',
              set_generator_factory=(
                  frequency_set_generator.HomogeneousMultiSetGenerator
                  .get_generator_factory_with_num_and_size(
                      universe_size=universe_size, num_sets=10,
                      set_size=set_size, freq_rates=[1]*10, freq_cap=3))),
          ScenarioConfig(
              name='heterogeneous',
              set_generator_factory=(
                  frequency_set_generator.HeterogeneousMultiSetGenerator
                  .get_generator_factory_with_num_and_size(
                      universe_size=universe_size, num_sets=10,
                      set_size=set_size, gamma_params=[[1, 1]]*10,
                      freq_cap=3))),
          ScenarioConfig(
              name='publisher_constant',
              set_generator_factory=(
                  frequency_set_generator.PublisherConstantFrequencySetGenerator
                  .get_generator_factory_with_num_and_size(
                      universe_size=universe_size, num_sets=10,
                      set_size=set_size, frequency=3))),
      )
    )


def _get_default_name_to_choices_of_set_size_list(
    small_set_size,
    large_set_size,
    num_sets
):
    return {
      'all_small': [small_set_size] * num_sets,
      'all_large': [large_set_size] * num_sets,
      '1st_small_then_large': (
          [small_set_size] + [large_set_size] * (num_sets - 1)),
      '1st_half_small_2nd_half_large': (
          [small_set_size] * int(num_sets / 2) +
          [large_set_size] * (num_sets - int(num_sets / 2))),
      'small_then_last_large': (
          [small_set_size] * (num_sets - 1) + [large_set_size]),
      'gradually_smaller': [
          int(large_set_size / np.sqrt(i + 1)) for i in range(num_sets)]
    }


def _generate_configs_scenario_1_2(universe_size, num_sets, small_set_size,
                                  large_set_size, remarketing_rate=None):
  """Generate configs of Scenario 1 & 2
  In this scenario,  publishers have heterogeneous users reach probability.
  The reach probability of a user in a publisher is the same as that in other
  publishers.

  If remarketing_rate is not provided, we take it as scenario 1 with universe_size
  as total. Othersize, we take it as scenario 2 with remarketing size as
  int(universe_size*remarketing_rate)

  See scenario 1 / 2:
  Independent m-publishers / n-publishers independently serve a remarketing list
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-1-independence-m-publishers-with-homogeneous-user-reach-probability
  Returns:
    A list of ScenarioConfigs of scenario 1 / 2 with selected parameters.
  """
  name_to_choices_of_set_size_list = _get_default_name_to_choices_of_set_size_list(
      small_set_size, large_set_size, num_sets
  )

  scenario_config_list = []
  if remarketing_rate is None:
    key_words = ['independent']
    size = universe_size
  else:
    size = int(universe_size * remarketing_rate)
    key_words = ['remarketing', 'remarketing_size:' + str(size)]

  for set_type, set_size_list in name_to_choices_of_set_size_list.items():
    scenario_config_list.append(
        ScenarioConfig(
            name='-'.join(key_words + [
                'universe_size:' + str(universe_size),
                'small_set:' + str(small_set_size),
                'large_set:' + str(large_set_size),
                'set_type:' + set_type]),
            set_generator_factory=(
                set_generator.IndependentSetGenerator
                .get_generator_factory_with_set_size_list(
                    universe_size=size,
                    set_size_list=set_size_list)))
    )

  return scenario_config_list


def _generate_configs_scenario_3(universe_size, num_sets, small_set_size,
                                  large_set_size, user_activity_assciation):
  """Generate configs of Scenario 3(a/b).

  In this scenario,  publishers have heterogeneous users reach probability.
  The reach probability of a user in a publisher is the same as that in other
  publishers.

  See scenario 3. [m-publishers with heterogeneous users reach probability] for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-3-m-publishers-with-heterogeneous-users-reach-probability

  Args:
    universe_size: the size of the pools from which the IDs will be selected.
    num_sets: the number of sets.
    small_set_size: the reach of the small reach sets.
    large_set_size: the reach of the large reach sets.
    user_activity_assciation: user activity association used in the Exponential
      Bow model. Should be one of the defined user activity association defined
      by the set_generator.USER_ACTIVITY_ASSOCIATION_XXX.
      For 3(a) user_activity_assciation=set_generator.USER_ACTIVITY_ASSOCIATION_INDEPENDENT
      3(b) USER_ACTIVITY_ASSOCIATION_IDENTICAL
  Returns:
    A list of ScenarioConfigs of scenario 3(a/b) with selected parameters.
  """
  name_to_choices_of_set_size_list = _get_default_name_to_choices_of_set_size_list(
      small_set_size, large_set_size, num_sets
  )

  scenario_config_list = []
  for set_type, set_size_list in name_to_choices_of_set_size_list.items():
    scenario_config_list.append(
        ScenarioConfig(
            name='-'.join([
                'exponential_bow',
                'user_activity_association:' + str(user_activity_assciation),
                'universe_size:' + str(universe_size),
                'small_set:' + str(small_set_size),
                'large_set:' + str(large_set_size),
                'set_type:' + set_type]),
            set_generator_factory=(
                set_generator.ExponentialBowSetGenerator
                .get_generator_factory_with_set_size_list(
                    user_activity_association=user_activity_assciation,
                    universe_size=universe_size,
                    set_size_list=set_size_list)))
    )
  return scenario_config_list


def _generate_configs_scenario_4a(universe_size, num_sets, small_set_size,
                                  large_set_size):
  """Generate configs of Scenario 4(a).

  In this setting, all the sets are identical.

  See Scenario 4: Full overlap or disjoint for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-4-full-overlap-or-disjoint

  Args:
    universe_size: the size of the pools from which the IDs will be selected.
    num_sets: the number of sets.
    small_set_size: the reach of the small reach sets.
    large_set_size: the reach of the large reach sets.

  Returns:
    A list of ScenarioConfigs of scenario 4(a) with either small or large
    reach.
  """
  scenario_config_list = []
  for set_size in [small_set_size, large_set_size]:
    scenario_config_list.append(
        ScenarioConfig(
            name='-'.join([
                'fully_overlapped',
                'universe_size:' + str(universe_size),
                'num_sets:' + str(num_sets),
                'set_sizes:' + str(set_size)
            ]),
            set_generator_factory=(
                set_generator.FullyOverlapSetGenerator
                .get_generator_factory_with_num_and_size(
                    universe_size=universe_size,
                    num_sets=num_sets,
                    set_size=set_size)))
    )
  return scenario_config_list


def _generate_configs_scenario_4b(universe_size, num_sets, small_set_size,
                                  large_set_size, order):
  """Generate configs of Scenario 4(b).

  In this scenario, sets are overlapped. That is, a small set is a subset of a
  large set.
  Currently only support a small set contained in a large set.
  Need to update set_generator.py to support more flexible set sizes.

  See Scenario 4: Full overlap or disjoint for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-4-full-overlap-or-disjoint


  Args:
    universe_size: the size of the pools from which the IDs will be selected.
    num_sets: the number of sets.
    small_set_size: the reach of the small reach sets.
    large_set_size: the reach of the large reach sets.
    order: the order of the sets. Should be one of set_generator.ORDER_XXX.

  Returns:
    A list of ScenarioConfigs of scenario 4(b) with subset sets.
  """
  scenario_config_list = []
  for num_large_sets in [1, int(num_sets / 2), num_sets - 1]:
    scenario_config_list.append(
        ScenarioConfig(
            name='-'.join([
                'subset',
                'universe_size:' + str(universe_size),
                'order:' + str(order),
                'num_large_sets:' + str(num_large_sets),
                'num_small_sets:' + str(num_sets - num_large_sets),
                'large_set_size:' + str(large_set_size),
                'small_set_size:' + str(small_set_size),
            ]),
            set_generator_factory=(
                set_generator.SubSetGenerator
                .get_generator_factory_with_num_and_size(
                    order=order,
                    universe_size=universe_size,
                    num_large_sets=num_large_sets,
                    num_small_sets=num_sets - num_large_sets,
                    large_set_size=large_set_size,
                    small_set_size=small_set_size)))
    )
  return scenario_config_list


def _generate_configs_scenario_5(num_sets, small_set_size,
                                 large_set_size, order, shared_prop_list):
  """Generate configs of Scenario 5.

  In this scenario, the sets are sequentially correlated.

  See Scenario 5: Sequentially correlated campaigns for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-5-sequentially-correlated-campaigns

  Args:
    num_sets: the number of sets.
    small_set_size: the reach of the small reach sets.
    large_set_size: the reach of the large reach sets.
    order: the order of the sets. Should be one of set_generator.ORDER_XXX.
    shared_prop_list: a sequence of the shared proportion of sequentially
      correlated sets.

  Returns:
    A list of ScenarioConfigs of scenario 5 Sequentiall correlated sets.
  """

  name_to_choices_of_set_size_list = {
      **_get_default_name_to_choices_of_set_size_list(
        small_set_size, large_set_size, num_sets
      ),
      'large_then_last_small': [large_set_size] * (num_sets - 1) + [
          small_set_size],
      'all_large_except_middle_small': (
          [large_set_size] * int(num_sets / 2) + [small_set_size]
          + [large_set_size] * (num_sets - 1 - int(num_sets / 2))),
      '1st_large_then_small': [large_set_size] + [small_set_size] * (
          num_sets - 1),
      'all_small_except_middle_large': (
          [small_set_size] * int(num_sets / 2) + [large_set_size]
          + [small_set_size] * (num_sets - 1 - int(num_sets / 2))),
      '1st_half_large_2nd_half_small': (
          [large_set_size] * int(num_sets / 2)
          + [small_set_size] * (num_sets - int(num_sets / 2))),
      'repeated_small_large': (
          [small_set_size, large_set_size] * int(num_sets / 2)
          + ([] if num_sets % 2 == 0 else [small_set_size]))
  }

  scenario_config_list = []
  for correlated_sets in (set_generator.CORRELATED_SETS_ONE,
                          set_generator.CORRELATED_SETS_ALL):
    for shared_prop in shared_prop_list:
      for set_type, set_size_list in name_to_choices_of_set_size_list.items():
        scenario_config_list.append(
            ScenarioConfig(
                name='-'.join([
                    'sequentially_correlated',
                    'order:' + str(order),
                    'correlated_sets:' + str(correlated_sets),
                    'shared_prop:' + str(shared_prop),
                    'set_type:' + str(set_type),
                    'large_set_size:' + str(large_set_size),
                    'small_set_size:' + str(small_set_size)
                ]),
                set_generator_factory=(
                    set_generator.SequentiallyCorrelatedSetGenerator.
                    get_generator_factory_with_set_size_list(
                        order=order,
                        correlated_sets=correlated_sets,
                        shared_prop=shared_prop,
                        set_size_list=set_size_list)))
        )
  return scenario_config_list


def _generate_freq_configs_scenario_1(universe_size, num_sets, set_size):
  """Generate configs of Frequency Scenario 1.

  See Frequency Scenario 1: Homogeneous user activities within a publisher for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-scenario-1-homogeneous-user-activities-within-a-publisher-1

  Args:
    universe_size: the universal size of reach
    num_sets: the number of sets
    set_size: size of each set, assuming they're all equal

  Returns:
    A list of ScenarioConfigs of freq scenario 1 Homogeneous user activities within a publisher correlated sets.
"""

  freq_rate_lists = [0.5, 1, 1.5, 2]
  freq_cap_lists = [3, 5, 10]
  scenario_config_list = []
  for freq_rate, freq_cap in itertools.product(freq_rate_lists, freq_cap_lists):
    scenario_config_list.append(
      ScenarioConfig(
        name='-'.join([
            'homogeneous',
            'universe_size:' + str(universe_size),
            'num_sets:' + str(num_sets),
            'freq_rate:' + str(freq_rate),
            'freq_cap:' + str(freq_cap),
        ]),
        set_generator_factory=(
            frequency_set_generator.HomogeneousMultiSetGenerator.
            get_generator_factory_with_num_and_size(
                universe_size=universe_size, num_sets=num_sets,
                set_size=set_size, freq_rates=[freq_rate]*num_sets,
                freq_cap=freq_cap
            ))),
      )
  return scenario_config_list


def _generate_freq_configs_scenario_2(universe_size, num_sets, set_size):
  """Generate configs of Frequency Scenario 2.

  See Frequency Scenario 2: Heterogeneous user frequency.:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-scenario-2-heterogeneous-user-frequency-1
  Args:
    universe_size: the universal size of reach
    num_sets: the number of sets
    set_size: size of each set, assuming they're all equal

  Returns:
    A list of ScenarioConfigs of freq scenario 2 heterogeneous user frequency.
"""

  distribution_rate_lists = [0.5, 1, 1.5, 2]
  freq_cap_lists = [3, 5, 10]
  scenario_config_list = []
  for distribution_rate, freq_cap in itertools.product(
      distribution_rate_lists, freq_cap_lists):
    scenario_config_list.append(
      ScenarioConfig(
        name='-'.join([
            'heterogeneous',
            'universe_size:' + str(universe_size),
            'num_sets:' + str(num_sets),
            'distribution_rate:' + str(distribution_rate),
            'freq_cap:' + str(freq_cap),
        ]),
        set_generator_factory=(
            frequency_set_generator.HeterogeneousMultiSetGenerator.
                get_generator_factory_with_num_and_size(
                    universe_size=universe_size, num_sets=num_sets,
                    set_size=set_size,
                    gamma_params=[[1,distribution_rate]]*num_sets,
                    freq_cap=freq_cap
            ))),
        )
  return scenario_config_list


def _generate_freq_configs_scenario_3(universe_size, num_sets, set_size):
  """Generate configs of Frequency Scenario 3.

  This is a stress testing, in which each publisher serves a fixed number of
  impressions to every reached id.

  See Frequency Scenario 3: Per-publisher frequency capping:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-scenario-3-per-publisher-frequency-capping

  Args:
    universe_size: the universal size of reach.
    num_sets: the number of sets.
    set_size: size of each set, assuming they're all equal.

  Returns:
    A list of ScenarioConfigs of freq scenario 3 per-publisher frequency
    capping.
  """
  frequency_list = [2, 3, 5, 10]
  scenario_config_list = []
  for frequency in frequency_list:
    scenario_config_list.append(
        ScenarioConfig(
            name='-'.join([
                'publisher_constant_frequency',
                'universe_size:' + str(universe_size),
                'num_sets:' + str(num_sets),
                'frequency:' + str(frequency),
            ]),
            set_generator_factory=(
                frequency_set_generator.PublisherConstantFrequencySetGenerator
                .get_generator_factory_with_num_and_size(
                    universe_size=universe_size,
                    num_sets=num_sets,
                    set_size=set_size,
                    frequency=frequency)
            )),
    )
  return scenario_config_list


def _complete_frequency_test_with_selected_parameters(
    num_runs=NUM_RUNS_VALUE,
    universe_size=FREQ_UNIVERSE_SIZE,
    num_sets=NUM_SETS_VALUE_FREQ,
    set_size=SET_SIZE_FOR_FREQ):
  """Generate configurations with handy selected parameters for scenarios.

  This evaluation covers the frequency simulation scenarios

  Args:
    num_runs: the number of runs per scenario * parameter setting.
    universe_size: the size of the pools from which the IDs will be selected.
    num_sets: the number of sets.
    set_size: reached size of each publisher, assuming all publishers have the
      same size

  Returns:
    An EvaluationConfig.
  """
  scenario_config_list = []
  # Scenario 1. Homogeneous user activities within a publisher
  scenario_config_list += _generate_freq_configs_scenario_1(
      universe_size, num_sets, set_size)
  # Scenario 2. Heterogeneous user frequency
  scenario_config_list += _generate_freq_configs_scenario_2(
      universe_size, num_sets, set_size)
  # Scenario 3. Per-publisher frequency capping.
  scenario_config_list += _generate_freq_configs_scenario_3(
      universe_size, num_sets, set_size)

  return EvaluationConfig(
    name='complete_frequency_test_with_selected_parameters',
    num_runs=num_runs,
    scenario_config_list=scenario_config_list)


def _complete_test_with_selected_parameters(
    num_runs=NUM_RUNS_VALUE,
    universe_size=UNIVERSE_SIZE_VALUE,
    num_sets=NUM_SETS_VALUE,
    order=set_generator.ORDER_RANDOM,
    small_set_size_rate=SMALL_REACH_RATE_VALUE,
    large_set_size_rate=LARGE_REACH_RATE_VALUE,
    remarketing_rate=REMARKETING_RATE_VALUE,
    shared_prop_list=SHARED_PROP_LIST_VALUE):
  """Generate configurations with handy selected parameters for scenarios.

  This evaluation covers the reach simulation scenarios

  Args:
    num_runs: the number of runs per scenario * parameter setting.
    universe_size: the size of the pools from which the IDs will be selected.
    num_sets: the number of sets.
    user_activity_assciation: user activity association used in the Exponential
      Bow model. Should be one of the defined user activity association defined
      by the set_generator.USER_ACTIVITY_ASSOCIATION_XXX.
    order: the order of the sets. Should be one of set_generator.ORDER_XXX.
    small_set_size_rate: the reach percentage of the small reach sets.
    large_set_size_rate: the reach percentage of the large reach sets.
    shared_prop_list: a sequence of the shared proportion of sequentially
      correlated sets.

  Returns:
    An EvaluationConfig.
  """
  scenario_config_list = []
  small_set_size = int(small_set_size_rate * universe_size)
  large_set_size = int(large_set_size_rate * universe_size)
  # Scenario 1. Independent publishers
  scenario_config_list += _generate_configs_scenario_1_2(
      universe_size, num_sets, small_set_size, large_set_size)
  # Scenario 2. publishers independently serve a remarketing list
  scenario_config_list += _generate_configs_scenario_1_2(
      universe_size, num_sets, small_set_size, large_set_size, remarketing_rate)

  # Scenario 3 (a). Exponential bow, independent user behavior.
  scenario_config_list += _generate_configs_scenario_3(
      universe_size, num_sets, small_set_size, large_set_size,
      set_generator.USER_ACTIVITY_ASSOCIATION_INDEPENDENT)

  # Scenario 3 (b). Exponential bow, identical user behavior.
  scenario_config_list += _generate_configs_scenario_3(
      universe_size, num_sets, small_set_size, large_set_size,
      set_generator.USER_ACTIVITY_ASSOCIATION_IDENTICAL)

  # Scenario 4(a). Fully-overlapped.
  scenario_config_list += _generate_configs_scenario_4a(
      universe_size, num_sets, small_set_size, large_set_size)

  # Scenario 4(b). Subset campaigns.
  scenario_config_list += _generate_configs_scenario_4b(
      universe_size, num_sets, small_set_size, large_set_size, order)

  # Scenario 5. Sequentially correlated campaigns
  scenario_config_list += _generate_configs_scenario_5(
      num_sets, small_set_size, large_set_size, order,
      shared_prop_list)

  return EvaluationConfig(
      name='complete_test_with_selected_parameters',
      num_runs=num_runs,
      scenario_config_list=scenario_config_list)


def _stress_test_cardinality_global_dp(universe_size=None,
                                       num_runs=NUM_RUNS_VALUE):
  """Stress test for cardinality estimator under global DP."""
  # The universe_size argument is included to conform to the run_evaluation
  # module.
  _ = universe_size
  scenario_config_list = []
  for scenario_id, reach in enumerate(sorted(REACH_STRESS_TEST)):
    scenario_config_list.append(ScenarioConfig(
        name=f'{scenario_id}-reach:{reach}',
        set_generator_factory=(
            set_generator.DisjointSetGenerator
            .get_generator_factory_with_set_size_list(
                set_sizes=[reach]))))
  return EvaluationConfig(
      name='global_dp_stress_test',
      num_runs=num_runs,
      scenario_config_list=scenario_config_list)


def _frequency_end_to_end_test(universe_size=10000, num_runs=NUM_RUNS_VALUE):
  """EvaluationConfig of end-to-end test of frequency evaluation code."""
  num_sets = 3
  set_size = int(universe_size * REACH_RATE_FREQ_END_TO_END_TEST)
  freq_rates = [1, 2, 3]
  freq_cap = 5
  return EvaluationConfig(
      name='frequency_end_to_end_test',
      num_runs=num_runs,
      scenario_config_list=[
          ScenarioConfig(
              name='-'.join([
                  'subset',
                  'universe_size:' + str(universe_size),
                  'num_sets:' + str(num_sets)
              ]),
              set_generator_factory=(
                  frequency_set_generator.HomogeneousMultiSetGenerator
                  .get_generator_factory_with_num_and_size(
                      universe_size=universe_size,
                      num_sets=num_sets,
                      set_size=set_size,
                      freq_rates=freq_rates,
                      freq_cap=freq_cap)))]
    )


def _generate_evaluation_configs():
  return (
      _smoke_test,
      _complete_test_with_selected_parameters,
      _stress_test_cardinality_global_dp,
      _frequency_end_to_end_test,
      _frequency_smoke_test,
      _complete_frequency_test_with_selected_parameters,
  )


def get_evaluation_config(config_name):
  """Returns the evaluation config with the specified config_name."""
  configs = _generate_evaluation_configs()
  valid_config_names = [c().name for c in configs]
  duplicate_configs = []
  for i in range(len(valid_config_names)-1):
    if valid_config_names[i] in valid_config_names[(i+1):]:
      duplicate_configs.append(valid_config_names[i])
  if duplicate_configs:
    raise ValueError("Duplicate names found in evaluation configs: {}".
                     format(','.join(duplicate_configs)))

  config = [c for c in configs if c().name == config_name]
  if not config:
    raise ValueError("Invalid evaluation config: {}\n"
                     "Valid choices are as follows: {}".format(
                     config_name, ','.join(valid_config_names)))
  return config[0]


def _format_epsilon(dp_type, epsilon=None, decimals=EPSILON_DECIMALS):
  """Format epsilon value to string.

  Args:
    dp_type: one of LOCAL_DP_STR and GLOBAL_DP_STR.
    epsilon: an optional differential private parameter. By default set to None.
    decimals: an integer value which set the number of decimal points of the
      epsilon to keep. By default, set to EPSILON_DECIMALS.

  Returns:
    A string representation of epsilon.

  Raises:
    ValueError: if dp_type is not one of 'local' and 'global'.
  """
  if epsilon is None:
    if dp_type == GLOBAL_DP_STR:
      return NO_GLOBAL_DP_STR
    elif dp_type == LOCAL_DP_STR:
      return NO_LOCAL_DP_STR
    else:
      raise ValueError(f'dp_type should be one of "{GLOBAL_DP_STR}" and '
                       f'"{LOCAL_DP_STR}".')

  str_format = dp_type + '_' + '{:0.' + str(decimals) + 'f}'
  return str_format.format(float(epsilon))


def _format_privacy_parameters(dp_type, epsilon=None, delta=None, num_queries=1,
                               noise_type=None,
                               epsilon_decimals=EPSILON_DECIMALS,
                               delta_decimals=DELTA_DECIMALS):
  """Format privacy parameters to string.

  Args:
    dp_type: one of LOCAL_DP_STR and GLOBAL_DP_STR.
    epsilon: an optional differential private parameter. By default set to None.
    delta: an optional differential private parameter. By default set to None.
      When delta is set, epsilon must also be set.
    num_queries: the number of queries over which the privacy budget is split.
    noise_type: the type of noise added. When set, must be one of GEOMETRIC_NOISE
      or GAUSSIAN_NOISE.
    epsilon_decimals: an integer value which set the number of decimal points of
      the epsilon to keep. By default, set to EPSILON_DECIMALS.
    delta_decimals: an integer value which set the number of decimal points of
      the delta to keep. By default, set to DELTA_DECIMALS.

  Returns:
    A string representation of the given privacy parameters.

  Raises:
    ValueError: if dp_type is not one of 'local' and 'global', or if delta is set
      without epsilon being set.
  """
  if epsilon is None:
    if delta is not None:
      raise ValueError(f'Delta cannot be set with epsilon unset: {delta}.')
    if dp_type == GLOBAL_DP_STR:
      return NO_GLOBAL_DP_STR
    elif dp_type == LOCAL_DP_STR:
      return NO_LOCAL_DP_STR
    else:
      raise ValueError(f'dp_type should be one of "{GLOBAL_DP_STR}" and '
                       f'"{LOCAL_DP_STR}".')

  epsilon_str = f'{epsilon:.{epsilon_decimals}f}'

  if delta is None:
    delta = 0
  delta_str = f'{delta:.{delta_decimals}f}'
    
  split_str = f'-budget_split-{num_queries}' if num_queries else ''
  
  noise_type_str = f'-{noise_type}' if noise_type else ''
    
  return (f'{dp_type}_{epsilon_str},{delta_str}{noise_type_str}{split_str}')

def construct_sketch_estimator_config_name(sketch_name, sketch_config,
                                           estimator_name, sketch_epsilon=None,
                                           estimate_epsilon=None,
                                           estimate_delta=None,
                                           num_estimate_queries=None,
                                           noise_type=None,
                                           max_frequency=None,
                                           epsilon_decimals=EPSILON_DECIMALS,
                                           delta_decimals=DELTA_DECIMALS):
  """Construct the name attribute for SketchEstimatorConfig.

  The name will be in the format of
  name_of_sketch-param_of_sketch-estimator_specification-sketch_epsilon
  -estimate_epsilon[-max_frequency].

  Args:
    sketch_name: a string of the estimator name. Do not include dash (-).
    sketch_config: a string of the sketch config. Do not include dash (-).
    estimator_name: a string of the estimator name.  Do not include dash (-).
    sketch_epsilon: an optional differential private parameter for the sketch.
      By default, set to None, i.e., not add noise to the sketch.
    estimate_epsilon: an optional differential private parameter for the
      estimate. By default, set to None, i.e., not add noise to the estimate.
    estimate_delta: an optional differential private parameter for the
      estimate. By default, set to None.
    num_estimate_queries: the number of queries over which the privacy budget
      for the estimate is split.
    noise_type: the type of noise added to each estimate. When set, must be one
      of GEOMETRIC_NOISE or GAUSSIAN_NOISE.
    max_frequency: an optional maximum frequency level. If not given, will not
      be added to the name.
    epsilon_decimals: an integer value which set the number of decimal points of
      the epsilon to keep. By default, set to EPSILON_DECIMALS.
    delta_decimals: an integer value which set the number of decimal points of
      the delta to keep. By default, set to DELTA_DECIMALS.

  Returns:
    The name of the SketchEstimatorConfig.

  Raises:
    AssertionError: if the input include dash (-).
  """
  for s in [sketch_name, sketch_config, estimator_name]:
    assert '-' not in s, f'Input should not contain "-", given {s}.'
  sketch_epsilon = _format_epsilon(
    LOCAL_DP_STR, epsilon=sketch_epsilon, decimals=epsilon_decimals)
  if num_estimate_queries is None:
    estimate_privacy_parameters = _format_epsilon(GLOBAL_DP_STR,
                                                  epsilon=estimate_epsilon,
                                                  decimals=epsilon_decimals)
  else:
    estimate_privacy_parameters = _format_privacy_parameters(
        GLOBAL_DP_STR, epsilon=estimate_epsilon, delta=estimate_delta,
        num_queries=num_estimate_queries, noise_type=noise_type,
        epsilon_decimals=epsilon_decimals, delta_decimals=delta_decimals)
  result = '-'.join([sketch_name, sketch_config, estimator_name, sketch_epsilon,
                     estimate_privacy_parameters])
  if max_frequency is not None:
    result = result + '-' + str(max_frequency)
  return result


# Document the estimators.
def _independent_set_estimator(sketch_epsilon=None, estimate_epsilon=None):
  """Generate a SketchEstimatorConfig for the independent set estimator.

  Use the Reach sketch as the underlying sketch. Set the universe size to
  UNIVERSE_SIZE_VALUE.

  Args:
    sketch_epsilon: a differential private parameter for the sketch.
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.

  Returns:
    A SketchEstimatorConfig for the independent estimator.
  """
  if sketch_epsilon:
    sketch_noiser = vector_of_counts.LaplaceNoiser(epsilon=sketch_epsilon)
  else:
    sketch_noiser = None

  if estimate_epsilon:
    estimate_noiser = estimator_noisers.LaplaceEstimateNoiser(
        epsilon=estimate_epsilon)
  else:
    estimate_noiser = None

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='reach_using_voc',
          sketch_config='1',
          estimator_name=f'independent_estimator_universe{UNIVERSE_SIZE_VALUE}',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon),
      sketch_factory=vector_of_counts.VectorOfCounts.get_sketch_factory(
          num_buckets=1),
      estimator=independent_set_estimator.IndependentSetEstimator(
          vector_of_counts.SequentialEstimator(), UNIVERSE_SIZE_VALUE),
      sketch_noiser=sketch_noiser,
      estimate_noiser=estimate_noiser
  )


def _hll_plus():
  """Generate a SketchEstimatorConfig for HyperLogLogPlus.

  Args:
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.

  Returns:
    A SketchEstimatorConfig for HyperLogLogPlus.
  """
  sketch_len = 2**14

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='hyper_log_log_plus',
          sketch_config=str(sketch_len),
          estimator_name='hll_cardinality'),
      sketch_factory=hyper_log_log.HyperLogLogPlusPlus.get_sketch_factory(
          length=sketch_len),
      estimator=hyper_log_log.HllCardinality(),
  )


def _log_bloom_filter_first_moment_log(length, sketch_epsilon=None,
                                       estimate_epsilon=None):
  """Generate a SketchEstimatorConfig for Log Bloom Filters.

  Args:
    length: the length of the log bloom filters.
    sketch_epsilon: a differential private parameter for the sketch.
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.

  Returns:
    A SketchEstimatorConfig for log Bloom Filters of length being 10**5.
  """
  if sketch_epsilon:
    sketch_noiser = bloom_filters.BlipNoiser(epsilon=sketch_epsilon)
    sketch_denoiser = bloom_filters.SurrealDenoiser(epsilon=sketch_epsilon)
  else:
    sketch_noiser, sketch_denoiser = None, None

  if estimate_epsilon:
    estimate_noiser = estimator_noisers.GeometricEstimateNoiser(
        epsilon=estimate_epsilon)
  else:
    estimate_noiser = None

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='log_bloom_filter',
          sketch_config=str(length),
          estimator_name='first_moment_log',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon),
      sketch_factory=bloom_filters.LogarithmicBloomFilter.get_sketch_factory(
          length=length),
      estimator=bloom_filters.FirstMomentEstimator(
          method=bloom_filters.FirstMomentEstimator.METHOD_LOG,
          noiser=estimate_noiser,
          denoiser=sketch_denoiser),
      sketch_noiser=sketch_noiser
  )


def _geo_bloom_filter_first_moment_geo(length,
                                       sketch_epsilon=None,
                                       estimate_epsilon=None):
  """Generate a SketchEstimatorConfig for Geometric Bloom Filters.

  The length of the Geometric Bloom Filters sketch will be set to 10**4 and the
  geometric distribution probability will be set to 0.0012.

  Args:
    sketch_epsilon: a differential private parameter for the sketch.

  Returns:
    A SketchEstimatorConfig for Geometric Bloom Filters of length being 10**4
    and geometric distribution probability being 0.0012.
  """
  if sketch_epsilon:
    sketch_noiser = bloom_filters.BlipNoiser(epsilon=sketch_epsilon)
    sketch_denoiser = bloom_filters.SurrealDenoiser(epsilon=sketch_epsilon)
  else:
    sketch_noiser, sketch_denoiser = None, None

  if estimate_epsilon:
    estimate_noiser = estimator_noisers.GeometricEstimateNoiser(
        epsilon=estimate_epsilon)
  else:
    estimate_noiser = None

  probability = GEO_LENGTH_PROB_PRODUCT / length
  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='geo_bloom_filter',
          sketch_config=f'{length}_{probability:.6f}',
          estimator_name='first_moment_geo',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon),
      sketch_factory=bloom_filters.GeometricBloomFilter.get_sketch_factory(
          length, probability),
      estimator=bloom_filters.FirstMomentEstimator(
          method=bloom_filters.FirstMomentEstimator.METHOD_GEO,
          noiser=estimate_noiser,
          denoiser=sketch_denoiser),
      sketch_noiser=sketch_noiser
  )


def _bloom_filter_first_moment_estimator_uniform(length, sketch_epsilon=None,
                                                 estimate_epsilon=None):
  """Generate a SketchEstimatorConfig for Bloom Filters.

  The bloom filter uses 1 hash functions.

  Args:
    length: the length of the bloom filter.
    sketch_epsilon: a differential private parameter for the sketch.
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.

  Returns:
    A SketchEstimatorConfig for Bloom Filters of with 1 hash function.
  """
  if sketch_epsilon:
    sketch_noiser = bloom_filters.BlipNoiser(epsilon=sketch_epsilon)
    sketch_denoiser = bloom_filters.SurrealDenoiser(epsilon=sketch_epsilon)
  else:
    sketch_noiser, sketch_denoiser = None, None

  if estimate_epsilon:
    estimate_noiser = estimator_noisers.GeometricEstimateNoiser(
        epsilon=estimate_epsilon)
  else:
    estimate_noiser = None

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='bloom_filter',
          sketch_config=str(length) + '_hash1',
          estimator_name='union_estimator',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon),
      sketch_factory=bloom_filters.UniformBloomFilter.get_sketch_factory(
          length),
      estimator=bloom_filters.FirstMomentEstimator(
          method=bloom_filters.FirstMomentEstimator.METHOD_UNIFORM,
          noiser=estimate_noiser,
          denoiser=sketch_denoiser),
      sketch_noiser=sketch_noiser
  )


def _exp_bloom_filter_first_moment_exp(length, sketch_epsilon=None,
                                       estimate_epsilon=None,
                                       estimate_delta=None,
                                       num_estimate_queries=None,
                                       noise_type=GEOMETRIC_NOISE,
                                        epsilon_decimals=EPSILON_DECIMALS):
  """Generate a SketchEstimatorConfig for Exponential Bloom Filters.

  The decay rate is 10.

  Args:
    length: the length of the exponential bloom filters.
    sketch_epsilon: a differential private parameter for the sketch.
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.
    estimate_delta: an optional differential private parameter for the
      estimate.
    num_estimate_queries: the number of queries over which the privacy budget
      for the estimate is split.
    noise_type: the type of noise added to each estimate. When noise is added,
      must be one of GEOMETRIC_NOISE, GAUSSIAN_NOISE or None.
    epsilon_decimals: an integer value which set the number of decimal
      points of the global epsilon to keep. By default, set to
      EPSILON_DECIMALS.

  Returns:
    A SketchEstimatorConfig for Exponential Bloom Filters of with decay rate
    being 10.

  Raises:
    ValueError: if estimate_epsilon is not None and noise_type is not one of
      GEOMETRIC_NOISE or GAUSSIAN_NOISE.
  """
  if sketch_epsilon:
    sketch_noiser = bloom_filters.BlipNoiser(epsilon=sketch_epsilon)
    sketch_denoiser = bloom_filters.SurrealDenoiser(epsilon=sketch_epsilon)
  else:
    sketch_noiser, sketch_denoiser = None, None

  if estimate_epsilon:
    if noise_type == GEOMETRIC_NOISE:
      if num_estimate_queries:
        estimate_epsilon_per_query = estimate_epsilon / num_estimate_queries
      else:
        estimate_epsilon_per_query = estimate_epsilon
      estimate_noiser = estimator_noisers.GeometricEstimateNoiser(
          estimate_epsilon_per_query)
    elif noise_type == GAUSSIAN_NOISE:
      estimate_noiser = estimator_noisers.GaussianEstimateNoiser(
          estimate_epsilon, estimate_delta, num_queries=num_estimate_queries)
    else:
      raise ValueError(f'noise_type should be one of "{GEOMETRIC_NOISE}" and '
                       f'"{GAUSSIAN_NOISE}".')
  else:
    estimate_noiser = None

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='exp_bloom_filter',
          sketch_config=str(length) + '_10',
          estimator_name='first_moment_exp',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon,
          estimate_delta=estimate_delta,
          num_estimate_queries=num_estimate_queries,
          noise_type=noise_type,
          epsilon_decimals=epsilon_decimals),
      sketch_factory=bloom_filters.ExponentialBloomFilter.get_sketch_factory(
          length=length, decay_rate=EXP_ADBF_DECAY_RATE),
      estimator=bloom_filters.FirstMomentEstimator(
          method=bloom_filters.FirstMomentEstimator.METHOD_EXP,
          noiser=estimate_noiser,
          denoiser=sketch_denoiser),
      sketch_noiser=sketch_noiser
  )

LIQUID_LEGIONS_1E5_10_LN3_SEQUENTIAL = SketchEstimatorConfig(
    name='liquid_legions-1e5_10-ln3-sequential',
    sketch_factory=liquid_legions.LiquidLegions.get_sketch_factory(
        a=10, m=10**5),
    estimator=liquid_legions.SequentialEstimator(),
    sketch_noiser=liquid_legions.Noiser(flip_probability=0.25))

LIQUID_LEGIONS_1E5_10_INFTY_SEQUENTIAL = SketchEstimatorConfig(
    name='liquid_legions-1e5_10-infty-sequential',
    sketch_factory=liquid_legions.LiquidLegions.get_sketch_factory(
        a=10, m=10**5),
    estimator=liquid_legions.SequentialEstimator())


def _vector_of_counts_4096_sequential(sketch_epsilon=None,
                                      estimate_epsilon=None):
  """Generate a SketchEstimatorConfig for Vector-of-Counts.

  The length of the Vector-of-Counts sketch will be set to 4096.

  Args:
    sketch_epsilon: a differential private parameter for the sketch.
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.

  Returns:
    A SketchEstimatorConfig for Vector-of-Counts of length being 4096.
  """
  if sketch_epsilon:
    sketch_noiser = vector_of_counts.LaplaceNoiser(epsilon=sketch_epsilon)
  else:
    sketch_noiser = None

  if estimate_epsilon:
    estimate_noiser = estimator_noisers.LaplaceEstimateNoiser(
        epsilon=estimate_epsilon)
  else:
    estimate_noiser = None

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='vector_of_counts',
          sketch_config='4096',
          estimator_name='sequential',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon),
      sketch_factory=vector_of_counts.VectorOfCounts.get_sketch_factory(
          num_buckets=4096),
      estimator=vector_of_counts.SequentialEstimator(),
      sketch_noiser=sketch_noiser,
      estimate_noiser=estimate_noiser
  )


def _meta_voc_for_exp_adbf(adbf_length, adbf_decay_rate, voc_length,
                           sketch_epsilon=None):
  """Construct Meta VoC estimator for the Exponential ADBF sketches.

  Args:
    adbf_length: the length of the Exp-ADBF sketch.
    adbf_decay_rate: the decay rate of the Exp-ADBF sketch.
    voc_length: the length of the VoC sketch.
    sketch_epsilon: the local DP epsilon value. By default, set to None,
      meaning that there won't be local noise used.

  Returns:
    A SketchEstimatorConfig for the Exp-ADBF using the Meta VoC estimator.
  """
  if sketch_epsilon is None:
    local_noiser = None
  else:
    local_noiser = vector_of_counts.LaplaceNoiser(epsilon=sketch_epsilon)

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='exp_bloom_filter',
          sketch_config=f'{adbf_length}_{adbf_decay_rate}',
          estimator_name=f'meta_voc_{voc_length}',
          sketch_epsilon=sketch_epsilon),
      sketch_factory=bloom_filters.ExponentialBloomFilter.get_sketch_factory(
          length=adbf_length, decay_rate=adbf_decay_rate),
      estimator=meta_estimators.MetaVectorOfCountsEstimator(
          num_buckets=int(voc_length),
          adbf_estimator=bloom_filters.FirstMomentEstimator(
              method=bloom_filters.FirstMomentEstimator.METHOD_EXP),
          meta_sketch_noiser=local_noiser,
      )
  )


def _meta_voc_for_bf(bf_length, voc_length, sketch_epsilon=None):
  """Construct Meta VoC estimator for the Bloom Filter sketches.

  Args:
    bf_length: the length of the Bloom Filter sketch.
    voc_length: the length of the VoC sketch.
    sketch_epsilon: the local DP epsilon value. By default, set to None,
      meaning that there won't be local noise used.

  Returns:
    A SketchEstimatorConfig for the Bloom Filter using the Meta VoC estimator.
  """
  if sketch_epsilon is None:
    local_noiser = None
  else:
    local_noiser = vector_of_counts.LaplaceNoiser(epsilon=sketch_epsilon)

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='bloom_filter',
          sketch_config=f'{bf_length}',
          estimator_name=f'meta_voc_{voc_length}',
          sketch_epsilon=sketch_epsilon),
      sketch_factory=(bloom_filters.UniformBloomFilter
                      .get_sketch_factory(length=int(bf_length))),
      estimator=meta_estimators.MetaVectorOfCountsEstimator(
          num_buckets=int(voc_length),
          adbf_estimator=bloom_filters.FirstMomentEstimator(
              method=bloom_filters.FirstMomentEstimator.METHOD_UNIFORM),
          meta_sketch_noiser=local_noiser,
      )
  )


def _generate_cardinality_estimator_configs():
  """Generate a tuple of cardinality estimator configs.

  Returns:
    A tuple of cardinality estimator configs.
  """
  configs = []

  # Construct configs for ADBFs with different lengths, sketch epsilon
  # and estimate epsilon.
  adbf_config_constructors = [
      _log_bloom_filter_first_moment_log,
      _exp_bloom_filter_first_moment_exp,
      _geo_bloom_filter_first_moment_geo,
  ]
  for config_constructor in adbf_config_constructors:
    for length in ADBF_LENGTH_LIST:
      for sketch_epsilon in SKETCH_EPSILON_VALUES:
        for estimate_epsilon in ESTIMATE_EPSILON_VALUES:
          configs.append(config_constructor(length, sketch_epsilon,
                                            estimate_epsilon))


  # Configs for testing global DP budget split
  for length in ADBF_LENGTH_LIST:
    for estimate_epsilon, estimate_delta in ESTIMATE_EPSILON_DELTA_VALUES:
      for num_estimate_queries in NUM_ESTIMATE_QUERIES_VALUES:
        for noise_type in [GAUSSIAN_NOISE, GEOMETRIC_NOISE]:
          configs.append(_exp_bloom_filter_first_moment_exp(
            length, estimate_epsilon=estimate_epsilon,
            estimate_delta=estimate_delta,
            num_estimate_queries=num_estimate_queries,
            noise_type=noise_type))

  # Construct configs for limit test under the global DP theme.
  for length in ADBF_LENGTH_LIST:
    for estimate_epsilon in GLOBAL_DP_LIMIT_TEST_EPSILON_VALUES:
      configs.append(_exp_bloom_filter_first_moment_exp(
          length, sketch_epsilon=None, estimate_epsilon=estimate_epsilon,
          epsilon_decimals=EPSILON_DECIMALS_LIMIT_TEST))


  # Configs of Vector-of-Counts.
  for sketch_epsilon in SKETCH_EPSILON_VALUES:
    for estimate_epsilon in ESTIMATE_EPSILON_VALUES:
      configs.append(_vector_of_counts_4096_sequential(sketch_epsilon,
                                                       estimate_epsilon))

  # Configs of independent estimator.
  for sketch_epsilon in SKETCH_EPSILON_VALUES:
    for estimate_epsilon in ESTIMATE_EPSILON_VALUES:
      configs.append(_independent_set_estimator(sketch_epsilon,
                                                estimate_epsilon))

  # Configs of hyper-log-log-plus.
  configs.append(_hll_plus())

  # Configs of Meta VoC for Exp-ADBF.
  for voc_length in VOC_LENGTH_LIST:
    for adbf_length in ADBF_LENGTH_LIST:
      for local_epsilon in SKETCH_EPSILON_VALUES:
        configs.append(_meta_voc_for_exp_adbf(
            adbf_length=adbf_length,
            adbf_decay_rate=EXP_ADBF_DECAY_RATE,
            voc_length=voc_length,
            sketch_epsilon=local_epsilon))

  # Configs of Meta VoC for BF.
  for voc_length in VOC_LENGTH_LIST:
    for bf_length in BLOOM_FILTERS_LENGTH_LIST:
      for local_epsilon in SKETCH_EPSILON_VALUES:
        configs.append(_meta_voc_for_bf(
            bf_length=bf_length,
            voc_length=voc_length,
            sketch_epsilon=local_epsilon))

  return tuple(configs)


def _stratiefied_sketch_vector_of_counts(max_frequency, clip, length,
                                         sketch_epsilon=None):
  """Construct configs of StratifiedSketch based on VectorOfCounts.

  Args:
    max_frequency: an integer indicating the maximum frequency to estimate.
    clip: a boolean indicating if or not to apply clipping for the
      Vector-of-Counts sketch.
    length: the length of Vector-of-Counts.
    sketch_epsilon: the DP epsilon for noising the Vector-of-Counts sketch.

  Returns:
    A SketchEstimatorConfig for stratified sketch with Vector-of-Counts as its
    base sketch.
  """
  if sketch_epsilon is not None:
    sketch_epsilon_float = sketch_epsilon
  else:
    sketch_epsilon_float = float('inf')
  sketch_operator = vector_of_counts_sketch_operator.StratifiedSketchOperator(
      clip=clip,
      epsilon=sketch_epsilon_float,
  )
  clip_str = 'clip' if clip else 'no_clip'
  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='stratified_sketch_vector_of_counts',
          sketch_config=str(length),
          estimator_name=f'sequential_{clip_str}',
          sketch_epsilon=sketch_epsilon,
          max_frequency=str(max_frequency)),
      sketch_factory=stratified_sketch.StratifiedSketch.get_sketch_factory(
          max_freq=max_frequency,
          cardinality_sketch_factory=(
              vector_of_counts.VectorOfCounts.get_sketch_factory(int(length))
          ),
          noiser_class=vector_of_counts.LaplaceNoiser,
          epsilon=sketch_epsilon_float,
          epsilon_split=0,
          union=sketch_operator.union,
      ),
      estimator=stratified_sketch.SequentialEstimator(
          sketch_operator=sketch_operator,
          cardinality_estimator=vector_of_counts.SequentialEstimator(
              clip=clip,
              epsilon=sketch_epsilon_float,
          ),
      ),
      max_frequency=max_frequency,
  )


def _stratiefied_sketch_geo_adbf(
    max_frequency, length, sketch_epsilon, global_epsilon,
    epsilon_split=STRATIFIED_EXP_ADBF_EPSILON_SPLIT):
  """Construct configs of StratifiedSketch based on geometric ADBF.

  Args:
    max_frequency: an integer indicating the maximum frequency to estimate.
    length: the length of geometric ADBF.
    sketch_epsilon: the DP epsilon for noising the geometric ADBF sketch.
    global_epsilon: the global DP epsilon parameter.
    epsilon_split : Ratio of privacy budget to spend to noise 1+ sketch. When
      epsilon_split=0 the 1+ sketch is created from the underlying exact set
      directly. epsilon_split should be smaller than 1.

  Returns:
    A SketchEstimatorConfig for stratified sketch with geometric ADBF as its
    base sketch.
  """
  if sketch_epsilon:
    sketch_epsilon_float = sketch_epsilon
    # The following denoiser is used by the cardinality estimator,
    # so the epsilon should be that after privacy budget (epsilon) splitting.
    sketch_denoiser = bloom_filters.SurrealDenoiser(
        epsilon=sketch_epsilon * epsilon_split)
  else:
    sketch_epsilon_float = float('inf')
    sketch_denoiser = None

  # Global noise.
  if global_epsilon is not None:
    estimate_noiser = estimator_noisers.GeometricEstimateNoiser(
        epsilon=global_epsilon)
  else:
    estimate_noiser = None

  sketch_operator = (
      bloom_filter_sketch_operators.ExpectationApproximationSketchOperator(
          estimation_method=bloom_filters.FirstMomentEstimator.METHOD_GEO))
  probability = GEO_LENGTH_PROB_PRODUCT / length

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='stratified_sketch_geo_adbf',
          sketch_config=f'{length}_{probability:.6f}',
          estimator_name='first_moment_estimator_geo_expectation',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=global_epsilon,
          max_frequency=str(max_frequency)),
      sketch_factory=stratified_sketch.StratifiedSketch.get_sketch_factory(
          max_freq=max_frequency,
          cardinality_sketch_factory=(
              bloom_filters.GeometricBloomFilter.get_sketch_factory(
                  length=length, probability=probability)
          ),
          noiser_class=bloom_filters.BlipNoiser,
          epsilon=sketch_epsilon_float,
          epsilon_split=epsilon_split,
          union=sketch_operator.union,
      ),
      estimator=stratified_sketch.SequentialEstimator(
          sketch_operator=sketch_operator,
          cardinality_estimator=bloom_filters.FirstMomentEstimator(
              method=bloom_filters.FirstMomentEstimator.METHOD_GEO,
              denoiser=sketch_denoiser,
              noiser=estimate_noiser,
          ),
      ),
      max_frequency=max_frequency,
  )


def _stratiefied_sketch_exponential_adbf(
    max_frequency, length, sketch_epsilon, global_epsilon,
    sketch_operator_type,
    epsilon_split=STRATIFIED_EXP_ADBF_EPSILON_SPLIT):
  """Construct configs of StratifiedSketch based on Exponential ADBF.

  Args:
    max_frequency: an integer indicating the maximum frequency to estimate.
    length: the length of Exponential ADBF.
    sketch_epsilon: the DP epsilon for noising the Exponential ADBF sketch.
    global_epsilon: the global DP epsilon parameter.
    sketch_operator_type: one of 'bayesian' and 'expectation'.
    epsilon_split : Ratio of privacy budget to spend to noise 1+ sketch. When
      epsilon_split=0 the 1+ sketch is created from the underlying exact set
      directly. epsilon_split should be smaller than 1.

  Returns:
    A SketchEstimatorConfig for stratified sketch with Exponential ADBF as its
    base sketch.

  Raises:
    ValueError: if the sketch_operator is not one of 'bayesian' and
    'expectation'.
  """
  # Local noise.
  if sketch_epsilon:
    sketch_epsilon_float = sketch_epsilon
    # The following denoiser is used by the cardinality estimator,
    # so the epsilon should be that after privacy budget (epsilon) splitting.
    sketch_denoiser = bloom_filters.SurrealDenoiser(
        epsilon=sketch_epsilon * epsilon_split)
  else:
    sketch_epsilon_float = float('inf')
    sketch_denoiser = None

  # Global noise.
  if global_epsilon is not None:
    estimate_noiser = estimator_noisers.GeometricEstimateNoiser(
        epsilon=global_epsilon)
  else:
    estimate_noiser = None

  if sketch_operator_type == SKETCH_OPERATOR_EXPECTATION:
    sketch_operator = (
        bloom_filter_sketch_operators.ExpectationApproximationSketchOperator(
            estimation_method=bloom_filters.FirstMomentEstimator.METHOD_EXP))
  elif sketch_operator_type == SKETCH_OPERATOR_BAYESIAN:
    sketch_operator = (
        bloom_filter_sketch_operators.BayesianApproximationSketchOperator(
            estimation_method=bloom_filters.FirstMomentEstimator.METHOD_EXP))
  else:
    raise ValueError('sketch operator should be one of '
                     '"{SKETCH_OPERATOR_BAYESIAN}" and '
                     '"{SKETCH_OPERATOR_EXPECTATION}".')

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='stratified_sketch_exp_adbf',
          sketch_config=f'{length}_{EXP_ADBF_DECAY_RATE}',
          estimator_name=f'first_moment_estimator_exp_{sketch_operator_type}',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=global_epsilon,
          max_frequency=str(max_frequency)),
      sketch_factory=stratified_sketch.StratifiedSketch.get_sketch_factory(
          max_freq=max_frequency,
          cardinality_sketch_factory=(
              bloom_filters.ExponentialBloomFilter.get_sketch_factory(
                  length=length, decay_rate=EXP_ADBF_DECAY_RATE)
          ),
          noiser_class=bloom_filters.BlipNoiser,
          epsilon=sketch_epsilon_float,
          epsilon_split=epsilon_split,
          union=sketch_operator.union,
      ),
      estimator=stratified_sketch.SequentialEstimator(
          sketch_operator=sketch_operator,
          cardinality_estimator=bloom_filters.FirstMomentEstimator(
              method=bloom_filters.FirstMomentEstimator.METHOD_EXP,
              denoiser=sketch_denoiser,
              noiser=estimate_noiser,
          ),
      ),
      max_frequency=max_frequency,
  )


def _exact_multi_set(max_frequency):
  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='exact_multi_set',
          sketch_config='10000',
          estimator_name='lossless',
          max_frequency=str(int(max_frequency))),
      sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
      estimator=exact_set.LosslessEstimator(),
      max_frequency=max_frequency,
  )


def _exp_same_key_aggregator(max_frequency, global_epsilon, length):
  """Create an ExponentialSameKeyAggregator config.

  Args:
    max_frequency: the maximum frequency to estimate.
    global_epsilon: the global DP epsilon parameter.
    length: the length of the ExponentialSameKeyAggregator.

  Returns:
    A SketchEstimatorConfig of ExponentialSameKeyAggregator.
  """
  if global_epsilon is not None:
    estimate_noiser_class = estimator_noisers.GeometricEstimateNoiser
  else:
    estimate_noiser_class = None

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='exp_same_key_aggregator',
          sketch_config='_'.join([str(int(length)), '10']),
          estimator_name='standardized_histogram',
          estimate_epsilon=global_epsilon,
          max_frequency=str(max_frequency)),
      sketch_factory=(
          same_key_aggregator.ExponentialSameKeyAggregator.get_sketch_factory(
              length, decay_rate=EXP_ADBF_DECAY_RATE)),
      estimator=same_key_aggregator.StandardizedHistogramEstimator(
          max_freq=max_frequency,
          noiser_class=estimate_noiser_class,
          epsilon=global_epsilon,
      ),
      max_frequency=max_frequency,
  )


def _generate_frequency_estimator_configs(max_frequency):
  """Create frequency estimator configurations."""
  configs = []

  # Stratified Sketch based on Vector-of-Counts.
  for epsilon, clip, length in itertools.product(SKETCH_EPSILON_VALUES,
                                                 [False, True],
                                                 VOC_LENGTH_LIST):
    configs.append(
        _stratiefied_sketch_vector_of_counts(max_frequency, clip, length,
                                             epsilon)
    )

  # Stratified Sketch based on exponential ADBF.
  for sketch_epsilon, global_epsilon, length, sketch_operator_type in (
      itertools.product(
          SKETCH_EPSILON_VALUES, ESTIMATE_EPSILON_VALUES, ADBF_LENGTH_LIST,
          SKETCH_OPERATOR_LIST)):
    configs.append(
        _stratiefied_sketch_exponential_adbf(max_frequency, length,
                                             sketch_epsilon, global_epsilon,
                                             sketch_operator_type)
    )

  for sketch_epsilon, global_epsilon, length in (
      itertools.product(
          SKETCH_EPSILON_VALUES, ESTIMATE_EPSILON_VALUES, ADBF_LENGTH_LIST)):
    configs.append(
        _stratiefied_sketch_geo_adbf(max_frequency, length,
                                     sketch_epsilon, global_epsilon)
    )
  # Exact set.
  configs.append(_exact_multi_set(max_frequency))

  # Same-key-aggregator.
  for global_epsilon, length in itertools.product(ESTIMATE_EPSILON_VALUES,
                                                  ADBF_LENGTH_LIST):
    configs.append(
        _exp_same_key_aggregator(max_frequency, global_epsilon, length))

  return tuple(configs)


def get_estimator_configs(estimator_names, max_frequency):
  """Returns a list of estimator configs by name.

  Args:
    estimator_names: a list of estimators defined in the evaluation_configs.
    max_frequency: an integer value of the maximum frequency level.

  Returns:
    A list of SketchEstimatorConfig.

  Raises:
    ValueError: if the estimator_names is not given, or any element of
      estimator_names is not defined in the evaluation_configs.
  """
  if not estimator_names:
    raise ValueError('No estimators were specified.')

  all_estimators = {
      conf.name: conf for conf in
      _generate_cardinality_estimator_configs()
      + _generate_frequency_estimator_configs(max_frequency)}
  estimator_list = [all_estimators[c] for c in estimator_names
                    if c in all_estimators]

  if len(estimator_list) == len(estimator_names):
    return estimator_list

  invalid_estimator_names = [c for c in estimator_names
                             if c not in all_estimators]

  raise ValueError('Invalid estimator(s): {}\nSupported estimators: {}'.
                   format(','.join(invalid_estimator_names),
                          ',\n'.join(all_estimators.keys())))
