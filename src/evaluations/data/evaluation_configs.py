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
import math

import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators import bloom_filters
from wfa_cardinality_estimation_evaluation_framework.estimators import estimator_noisers
from wfa_cardinality_estimation_evaluation_framework.estimators import exact_set
from wfa_cardinality_estimation_evaluation_framework.estimators import independent_set_estimator
from wfa_cardinality_estimation_evaluation_framework.estimators import liquid_legions
from wfa_cardinality_estimation_evaluation_framework.estimators import vector_of_counts
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
UNIVERSE_SIZE_VALUE = 100000
NUM_SETS_VALUE = 20
SMALL_REACH_RATE_VALUE = 0.01
LARGE_REACH_RATE_VALUE = 0.2
SHARED_PROP_LIST_VALUE = (0.25, 0.5, 0.75)
REMARKETING_RATE_VALUE = 0.2

NO_GLOBAL_DP_STR = 'no_global_dp'
GLOBAL_DP_STR = 'global_dp'
NO_LOCAL_DP_STR = 'no_local_dp'
LOCAL_DP_STR = 'local_dp'

# The None in the epsilon value is used to tell the sketch estimator constructor
# that we do not want to noise the sketch.
SKETCH_EPSILON_VALUES = (math.log(3), math.log(3) / 4, math.log(3) / 10, None)
# The current simulator module add noise to the estimated cardinality so as to
# mimic the global differential privacy use case. In the real world, the
# implementation could be different and more complicated.
# As such, we use a small epsilon so as to be conservative on the result.
ESTIMATE_EPSILON_VALUES = (math.log(3), None)

# The length of the Any Distribution Bloom Filters.
# We use the np.array with dtype so as to make sure that the lengths are all
# integers.
ADBF_LENGTH_LIST = np.array([1e5, 2.5e5], dtype=np.int64)

# The length of the bloom filters.
BLOOM_FILTERS_LENGTH_LIST = np.array([5e6], dtype=np.int64)


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
                      num_sets=20, set_size=10000,
                      shared_prop=0.5))),
          ScenarioConfig(
              name='sequentially_correlated_one',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory_with_num_and_size(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ONE,
                      num_sets=20, set_size=10000,
                      shared_prop=0.5))),
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


def _frequency_end_to_end_test(num_runs=NUM_RUNS_VALUE):
  """EvaluationConfig of end-to-end test of frequency evaluation code."""
  num_sets = 3
  universe_size = 10000
  set_size = 5000
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
      _frequency_end_to_end_test
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


def _format_epsilon(dp_type, epsilon=None, decimals=4):
  """Format epsilon value to string.

  Args:
    dp_type: one of LOCAL_DP_STR and GLOBAL_DP_STR.
    epsilon: an optional differential private parameter. By default set to None.
    decimals: an integer value which set the number of decimal points of the
      epsilon to keep.

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


def construct_sketch_estimator_config_name(sketch_name, sketch_config,
                                           estimator_name, sketch_epsilon=None,
                                           estimate_epsilon=None,
                                           max_frequency=None):
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
    max_frequency: an optional maximum frequency level. If not given, will not
      be added to the name.

  Returns:
    The name of the SketchEstimatorConfig.

  Raises:
    AssertionError: if the input include dash (-).
  """
  for s in [sketch_name, sketch_config, estimator_name]:
    assert '-' not in s, f'Input should not contain "-", given {s}.'
  sketch_epsilon = _format_epsilon(LOCAL_DP_STR, sketch_epsilon)
  estimate_epsilon = _format_epsilon(GLOBAL_DP_STR, estimate_epsilon)
  result = '-'.join([sketch_name, sketch_config, estimator_name, sketch_epsilon,
                     estimate_epsilon])
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


def _geo_bloom_filter_first_moment_geo(sketch_epsilon=None):
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

  return SketchEstimatorConfig(
      name=construct_sketch_estimator_config_name(
          sketch_name='geo_bloom_filter',
          sketch_config='1e4_0.0012',
          estimator_name='first_moment_geo',
          sketch_epsilon=sketch_epsilon),
      sketch_factory=bloom_filters.GeometricBloomFilter.get_sketch_factory(
          10000, 0.0012),
      estimator=bloom_filters.FirstMomentEstimator(
          method=bloom_filters.FirstMomentEstimator.METHOD_GEO,
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
                                       estimate_epsilon=None):
  """Generate a SketchEstimatorConfig for Exponential Bloom Filters.

  The decay rate is 10.

  Args:
    length: the length of the exponential bloom filters.
    sketch_epsilon: a differential private parameter for the sketch.
    estimate_epsilon: a differential private parameter for the estimated
      cardinality.

  Returns:
    A SketchEstimatorConfig for Exponential Bloom Filters of with decay rate
    being 10.
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
          sketch_name='exp_bloom_filter',
          sketch_config=str(length) + '_10',
          estimator_name='first_moment_exp',
          sketch_epsilon=sketch_epsilon,
          estimate_epsilon=estimate_epsilon),
      sketch_factory=bloom_filters.ExponentialBloomFilter.get_sketch_factory(
          length=length, decay_rate=10),
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
  ]
  for config_constructor in adbf_config_constructors:
    for length in ADBF_LENGTH_LIST:
      for sketch_epsilon in SKETCH_EPSILON_VALUES:
        for estimate_epsilon in ESTIMATE_EPSILON_VALUES:
          configs.append(config_constructor(length, sketch_epsilon,
                                            estimate_epsilon))

  # Construct configs for estimators that currently doesn't support global DP.
  for sketch_epsilon in SKETCH_EPSILON_VALUES:
    configs.append(_geo_bloom_filter_first_moment_geo(sketch_epsilon))

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

  return tuple(configs)


def _generate_frequency_estimator_configs(max_frequency):
  return (
      SketchEstimatorConfig(
          name=construct_sketch_estimator_config_name(
              sketch_name='exact_multi_set',
              sketch_config='10000',
              estimator_name='lossless',
              max_frequency=str(int(max_frequency))),
          sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
          estimator=exact_set.LosslessEstimator(),
          max_frequency=max_frequency),
      )


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
