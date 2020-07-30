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
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators import bloom_filters
from wfa_cardinality_estimation_evaluation_framework.estimators import liquid_legions
from wfa_cardinality_estimation_evaluation_framework.estimators import vector_of_counts
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import EvaluationConfig
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import ScenarioConfig
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import SketchEstimatorConfig

SKETCH = 'sketch'
SKETCH_CONFIG = 'sketch_config'
EPSILON = 'epsilon'
ESTIMATOR = 'estimator'

SKETCH_ESTIMATOR_CONFIG_NAMES_FORMAT = (
    SKETCH, SKETCH_CONFIG, EPSILON, ESTIMATOR)

NUM_RUNS_VALUE = 100
UNIVERSE_SIZE_VALUE = 100000
NUM_SETS_VALUE = 20
SMALL_REACH_RATE_VALUE = 0.01
LARGE_REACH_RATE_VALUE = 0.2
SHARED_PROP_LIST_VALUE = (0.25, 0.5, 0.75)


# Document the evaluation configurations.
def _smoke_test(num_runs=NUM_RUNS_VALUE):
  """Smoke test evaluation configurations.

  We set the smoke test parameters according to Appendix 3: Example
  Parameters of Scenarios of the Cardinality and Frequency Estimation
  Evaluation Framework.

  Args:
    num_runs: the number of runs per scenario.

  Returns:
    An EvaluationConfig.
  """
  return EvaluationConfig(
      name='smoke_test',
      num_runs=num_runs,
      scenario_config_list=(
          ScenarioConfig(
              name='independent',
              set_generator_factory=(
                  set_generator.IndependentSetGenerator.
                  get_generator_factory_with_num_and_size(
                      universe_size=200000, num_sets=20, set_size=20000))),
          ScenarioConfig(
              name='remarketing',
              set_generator_factory=(
                  set_generator.IndependentSetGenerator.
                  get_generator_factory_with_num_and_size(
                      universe_size=40000, num_sets=20, set_size=20000))),
          ScenarioConfig(
              name='fully_overlapping',
              set_generator_factory=(
                  set_generator.FullyOverlapSetGenerator.
                  get_generator_factory_with_num_and_size(
                      universe_size=200000, num_sets=20, set_size=20000))),
          ScenarioConfig(
              name='sequentially_correlated_all',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory_with_num_and_size(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ALL,
                      universe_size=3 * 10**8, num_sets=20, set_size=10000,
                      shared_prop=0.5))),
          ScenarioConfig(
              name='sequentially_correlated_one',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory_with_num_and_size(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ONE,
                      universe_size=3 * 10**8, num_sets=20, set_size=10000,
                      shared_prop=0.5))),
          )
      )


def _generate_configs_scenario_3b(universe_size, num_sets, small_set_size,
                                  large_set_size, user_activity_assciation):
  """Generate configs of Scenario 3(b).

  In this scenario,  publishers have heterogeneous users reach probability.
  The reach probability of a user in a publisher is the same as that in other
  publishers.

  See section (b). [fully correlated user reach probabilities] for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-3-m-publishers-with-heterogeneous-users-reach-probability

  Args:
    universe_size: the size of the pools from which the IDs will be selected.
    num_sets: the number of sets.
    small_set_size: the reach of the small reach sets.
    large_set_size: the reach of the large reach sets.
    user_activity_assciation: user activity association used in the Exponential
      Bow model. Should be one of the defined user activity association defined
      by the set_generator.USER_ACTIVITY_ASSOCIATION_XXX.

  Returns:
    A list of ScenarioConfigs of scenario 3(b) with selected parameters.
  """
  name_to_choices_of_set_size_list = {
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
          large_set_size / np.sqrt(i + 1) for i in range(num_sets)]
  }

  scenario_config_list = []
  for set_size_name, set_size_list in name_to_choices_of_set_size_list.items():
    scenario_config_list.append(
        ScenarioConfig(
            name='-'.join([
                'exponential_bow',
                'user_activity_association:' + str(user_activity_assciation),
                'universe_size:' + str(universe_size),
                'small_set:' + str(small_set_size),
                'large_set:' + str(large_set_size),
                'set_sizes:' + set_size_name]),
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


def _generate_configs_scenario_5(universe_size, num_sets, small_set_size,
                                 large_set_size, order, shared_prop_list):
  """Generate configs of Scenario 5.

  In this scenario, the sets are sequentially correlated.

  See Scenario 5: Sequentially correlated campaigns for more details:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#scenario-5-sequentially-correlated-campaigns

  Args:
    universe_size: the size of the pools from which the IDs will be selected.
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
      '1st_small_then_large': [small_set_size] + [large_set_size] * (
          num_sets - 1),
      'large_then_last_small': [large_set_size] * (num_sets - 1) + [
          small_set_size],
      'all_large_except_middle_small': (
          [large_set_size] * int(num_sets / 2) + [small_set_size]
          + [large_set_size] * (num_sets - 1 - int(num_sets / 2))),
      '1st_large_then_small': [large_set_size] + [small_set_size] * (
          num_sets - 1),
      'small_then_last_large': [small_set_size] * (num_sets - 1) + [
          large_set_size],
      'all_small_except_middle_large': (
          [small_set_size] * int(num_sets / 2) + [large_set_size]
          + [small_set_size] * (num_sets - 1 - int(num_sets / 2))),
      '1st_half_small_2nd_half_large': (
          [small_set_size] * int(num_sets / 2)
          + [large_set_size] * (num_sets - int(num_sets / 2))),
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
      for set_name, set_size_list in name_to_choices_of_set_size_list.items():
        scenario_config_list.append(
            ScenarioConfig(
                name='-'.join([
                    'sequentially_correlated',
                    'universe_size:' + str(universe_size),
                    'order:' + str(order),
                    'correlated_sets:' + str(correlated_sets),
                    'shared_prop:' + str(shared_prop),
                    'set_sizes:' + str(set_name),
                    'large_set_size:' + str(large_set_size),
                    'small_set_size:' + str(small_set_size)
                ]),
                set_generator_factory=(
                    set_generator.SequentiallyCorrelatedSetGenerator.
                    get_generator_factory_with_set_size_list(
                        order=order,
                        correlated_sets=correlated_sets,
                        universe_size=universe_size,
                        shared_prop=shared_prop,
                        set_size_list=set_size_list)))
        )
  return scenario_config_list


def _complete_test_with_selected_parameters(
    num_runs=NUM_RUNS_VALUE,
    universe_size=UNIVERSE_SIZE_VALUE,
    num_sets=NUM_SETS_VALUE,
    user_activity_assciation=set_generator.USER_ACTIVITY_ASSOCIATION_IDENTICAL,
    order=set_generator.ORDER_RANDOM,
    small_set_size_rate=SMALL_REACH_RATE_VALUE,
    large_set_size_rate=LARGE_REACH_RATE_VALUE,
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

  # Scenario 3 (b). Exponential bow, identical user behavior.
  scenario_config_list.append(
      _generate_configs_scenario_3b(
          universe_size, num_sets, small_set_size, large_set_size,
          user_activity_assciation))

  # Scenario 4(a). Fully-overlapped.
  scenario_config_list.append(
      _generate_configs_scenario_4a(
          universe_size, num_sets, small_set_size, large_set_size))

  # Scenario 4(b). Subset campaigns.
  scenario_config_list.append(
      _generate_configs_scenario_4b(
          universe_size, num_sets, small_set_size, large_set_size, order))

  # Scenario 5. Sequantially correlated campaigns
  scenario_config_list.append(
      _generate_configs_scenario_5(
          universe_size, num_sets, small_set_size, large_set_size, order,
          shared_prop_list))

  return EvaluationConfig(
      name='complete_test_with_selected_parameters',
      num_runs=num_runs,
      scenario_config_list=scenario_config_list)


EVALUATION_CONFIGS_TUPLE = (
    _smoke_test,
    _complete_test_with_selected_parameters,
)


NAME_TO_EVALUATION_CONFIGS = {
    conf().name: conf for conf in EVALUATION_CONFIGS_TUPLE
}

EVALUATION_CONFIG_NAMES = tuple(NAME_TO_EVALUATION_CONFIGS.keys())


# Document the estimators.
# The name attribute of the SketchEstimatorConfig should conform to
# name_of_sketch-param_of_sketch-epsilon_value-estimator_specification.
# For example, if a user want to evaluate Bloom Filter of length 1000 with
# epsilon 0.1, and the UnionEstimator, then the name could be:
# bloom_filter-1e4-0.1-union.
LOG_BLOOM_FILTER_1E5_LN3_FIRST_MOMENT_LOG = SketchEstimatorConfig(
    name='log_bloom_filter-1e5-ln3-first_moment_log',
    sketch_factory=bloom_filters.LogarithmicBloomFilter.get_sketch_factory(
        length=10**5),
    estimator=bloom_filters.FirstMomentEstimator(
        method=bloom_filters.FirstMomentEstimator.METHOD_LOG,
        denoiser=bloom_filters.SurrealDenoiser(probability=0.25)),
    sketch_noiser=bloom_filters.BlipNoiser(epsilon=np.log(3)))

LOG_BLOOM_FILTER_1E5_INFTY_FIRST_MOMENT_LOG = SketchEstimatorConfig(
    name='log_bloom_filter-1e5-infty-first_moment_log',
    sketch_factory=bloom_filters.LogarithmicBloomFilter.get_sketch_factory(
        length=10**5),
    estimator=bloom_filters.FirstMomentEstimator(
        method=bloom_filters.FirstMomentEstimator.METHOD_LOG))

EXP_BLOOM_FILTER_1E5_10_LN3_FIRST_MOMENT_LOG = SketchEstimatorConfig(
    name='exp_bloom_filter-1e5_10-ln3-first_moment_exp',
    sketch_factory=bloom_filters.ExponentialBloomFilter.get_sketch_factory(
        length=10**5, decay_rate=10),
    estimator=bloom_filters.FirstMomentEstimator(
        method=bloom_filters.FirstMomentEstimator.METHOD_EXP,
        denoiser=bloom_filters.SurrealDenoiser(probability=0.25)),
    sketch_noiser=bloom_filters.BlipNoiser(epsilon=np.log(3)))

EXP_BLOOM_FILTER_1E5_10_INFTY_FIRST_MOMENT_LOG = SketchEstimatorConfig(
    name='exp_bloom_filter-1e5_10-infty-first_moment_exp',
    sketch_factory=bloom_filters.ExponentialBloomFilter.get_sketch_factory(
        length=10**5, decay_rate=10),
    estimator=bloom_filters.FirstMomentEstimator(
        method=bloom_filters.FirstMomentEstimator.METHOD_EXP))

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

VECTOR_OF_COUNTS_4096_LN3_SEQUENTIAL = SketchEstimatorConfig(
    name='vector_of_counts-4096-ln3-sequential',
    sketch_factory=vector_of_counts.VectorOfCounts.get_sketch_factory(
        num_buckets=4096),
    estimator=vector_of_counts.SequentialEstimator(),
    sketch_noiser=vector_of_counts.LaplaceNoiser(epsilon=np.log(3)))

VECTOR_OF_COUNTS_4096_INFTY_SEQUENTIAL = SketchEstimatorConfig(
    name='vector_of_counts-4096-infty-sequential',
    sketch_factory=vector_of_counts.VectorOfCounts.get_sketch_factory(
        num_buckets=4096),
    estimator=vector_of_counts.SequentialEstimator())

SKETCH_ESTIMATOR_CONFIGS_TUPLE = (
    LOG_BLOOM_FILTER_1E5_LN3_FIRST_MOMENT_LOG,
    LOG_BLOOM_FILTER_1E5_INFTY_FIRST_MOMENT_LOG,
    EXP_BLOOM_FILTER_1E5_10_LN3_FIRST_MOMENT_LOG,
    EXP_BLOOM_FILTER_1E5_10_INFTY_FIRST_MOMENT_LOG,
    LIQUID_LEGIONS_1E5_10_LN3_SEQUENTIAL,
    LIQUID_LEGIONS_1E5_10_INFTY_SEQUENTIAL,
    VECTOR_OF_COUNTS_4096_LN3_SEQUENTIAL,
    VECTOR_OF_COUNTS_4096_INFTY_SEQUENTIAL)

NAME_TO_ESTIMATOR_CONFIGS = {
    conf.name: conf for conf in SKETCH_ESTIMATOR_CONFIGS_TUPLE}

ESTIMATOR_CONFIG_NAMES = tuple(NAME_TO_ESTIMATOR_CONFIGS.keys())
