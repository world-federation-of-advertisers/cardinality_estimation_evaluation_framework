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
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import SketchEstimatorConfig
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


# Document the evaluation configurations.
def _smoke_test(num_runs=100):
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
                  set_generator.IndependentSetGenerator.get_generator_factory(
                      universe_size=200000, num_sets=20, set_size=20000))),
          ScenarioConfig(
              name='remarketing',
              set_generator_factory=(
                  set_generator.IndependentSetGenerator.get_generator_factory(
                      universe_size=40000, num_sets=20, set_size=20000))),
          ScenarioConfig(
              name='fully_overlapping',
              set_generator_factory=(
                  set_generator.FullyOverlapSetGenerator.get_generator_factory(
                      universe_size=200000, num_sets=20, set_size=20000))),
          ScenarioConfig(
              name='sequentially_correlated_all',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ALL,
                      universe_size=3 * 10**8, num_sets=20, set_size=10000,
                      shared_prop=0.5))),
          ScenarioConfig(
              name='sequentially_correlated_one',
              set_generator_factory=(
                  set_generator.SequentiallyCorrelatedSetGenerator
                  .get_generator_factory(
                      order=set_generator.ORDER_ORIGINAL,
                      correlated_sets=set_generator.CORRELATED_SETS_ONE,
                      universe_size=3 * 10**8, num_sets=20, set_size=10000,
                      shared_prop=0.5))),
          )
      )


def _test_with_selected_parameters_second_part(num_runs=100):
  """Configurations with handy selected parameters for scenarios 3b - 5.

  Args:
    num_runs: the number of runs per scenario * parameter setting.

  Returns:
    An EvaluationConfig.
  """
  scenario_config_list = []
  # Common parameters
  universe_size = 100000
  # Now fix the universe size. Can vary it if necessary.
  num_sets = 20
  order = set_generator.ORDER_RANDOM
  # Apply random order of sets here, assuming that we shuffle the publishers
  # before the estimation.
  small_reach_rate = 0.01
  large_reach_rate = 0.2
  small_reach = int(small_reach_rate * universe_size)
  large_reach = int(large_reach_rate * universe_size)

  # Scenario 3 (b). Exponential bow, identical user behavior
  user_activity_assciation = set_generator.USER_ACTIVITY_ASSOCIATION_IDENTICAL
  choises_of_set_size_list = (
      [small_reach] * num_sets,
      [large_reach] * num_sets,
      [small_reach] + [large_reach] * (num_sets - 1),
      [small_reach] * int(num_sets / 2) +
      [large_reach] * (num_sets - int(num_sets / 2)),
      [small_reach] * (num_sets - 1) + [large_reach],
      [large_reach / np.sqrt(i + 1) for i in range(num_sets)]
  )
  for set_size_list in choises_of_set_size_list:
    scenario_config_list.append(
        ScenarioConfig(
            name='exponential_bow',
            set_generator_factory=(
                set_generator.ExponentialBowSetGenerator.
                get_generator_factory_with_set_size_list(
                    user_activity_association=user_activity_assciation,
                    universe_size=universe_size,
                    set_size_list=set_size_list)))
    )

  # Scenario 4(a). Fully overlapped
  for set_size in [small_reach, large_reach]:
    scenario_config_list.append(
        ScenarioConfig(
            name='fully_overlapped',
            set_generator_factory=(
                set_generator.FullyOverlapSetGenerator.
                get_generator_factory_with_num_and_size(
                    universe_size=universe_size,
                    num_sets=num_sets,
                    set_size=set_size)))
    )

  # Scenario 4(b). Subset campaigns
  # Currently only support a small set contained in a large set.
  # Need to update set_generator.py to support more flexible set sizes.
  for num_large_sets in [1, 10, 19]:
    scenario_config_list.append(
        ScenarioConfig(
            name='subset',
            set_generator_factory=(
                set_generator.SubSetGenerator.
                get_generator_factory_with_num_and_size(
                    order=order,
                    universe_size=universe_size,
                    num_large_sets=num_large_sets,
                    num_small_sets=num_sets - num_large_sets,
                    large_set_size=large_reach,
                    small_set_size=small_reach)))
    )

  # Scenario 5. Sequantially correlated campaigns
  choices_of_set_size_list = [
      [small_reach] + [large_reach] * (num_sets - 1),
      [large_reach] * (num_sets - 1) + [small_reach],
      [large_reach] * int(num_sets / 2) + [small_reach] +
      [large_reach] * (num_sets - 1 - int(num_sets / 2)),
      [large_reach] + [small_reach] * (num_sets - 1),
      [small_reach] * (num_sets - 1) + [large_reach],
      [small_reach] * int(num_sets / 2) + [large_reach] +
      [small_reach] * (num_sets - 1 - int(num_sets / 2)),
      [small_reach] * int(num_sets / 2) +
      [large_reach] * (num_sets - int(num_sets / 2)),
      [large_reach] * int(num_sets / 2) +
      [small_reach] * (num_sets - int(num_sets / 2)),
      [small_reach, large_reach] * int(num_sets / 2) +
      ([] if num_sets % 2 == 0 else [small_reach])
  ]
  choices_of_shared_prop = [0.25, 0.5, 0.75]

  for correlated_sets in (set_generator.CORRELATED_SETS_ONE,
                          set_generator.CORRELATED_SETS_ALL):
    for shared_prop in choices_of_shared_prop:
      for set_size_list in choices_of_set_size_list:
        scenario_config_list.append(
            ScenarioConfig(
                name='sequentially_correlated',
                set_generator_factory=(
                    set_generator.SequentiallyCorrelatedSetGenerator.
                    get_generator_factory_with_set_size_list(
                        order=order,
                        correlated_sets=correlated_sets,
                        universe_size=universe_size,
                        shared_prop=shared_prop,
                        set_size_list=set_size_list)))
        )

  return EvaluationConfig(
      name='test_with_selected_parameters_second_part',
      num_runs=num_runs,
      scenario_config_list=scenario_config_list)


EVALUATION_CONFIGS_TUPLE = (
    _smoke_test,
    _test_with_selected_parameters_second_part,
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
    noiser=bloom_filters.BlipNoiser(epsilon=np.log(3)))

LOG_BLOOM_FILTER_1E5_0_FIRST_MOMENT_LOG = SketchEstimatorConfig(
    name='log_bloom_filter-1e5-0-first_moment_log',
    sketch_factory=bloom_filters.LogarithmicBloomFilter.get_sketch_factory(
        length=10**5),
    estimator=bloom_filters.FirstMomentEstimator(
        method=bloom_filters.FirstMomentEstimator.METHOD_LOG),
    noiser=None)

LIQUID_LEGIONS_1E5_10_LN3_SEQUENTIAL = SketchEstimatorConfig(
    name='liquid_legions-1e5_10-ln3-sequential',
    sketch_factory=liquid_legions.LiquidLegions.get_sketch_factory(
        a=10, m=10**5),
    estimator=liquid_legions.SequentialEstimator(),
    noiser=liquid_legions.Noiser(flip_probability=0.25))

LIQUID_LEGIONS_1E5_10_0_SEQUENTIAL = SketchEstimatorConfig(
    name='liquid_legions-1e5_10-0-sequential',
    sketch_factory=liquid_legions.LiquidLegions.get_sketch_factory(
        a=10, m=10**5),
    estimator=liquid_legions.SequentialEstimator(),
    noiser=None)

VECTOR_OF_COUNTS_4096_LN3_SEQUENTIAL = SketchEstimatorConfig(
    name='vector_of_counts-4096-ln3-sequential',
    sketch_factory=vector_of_counts.VectorOfCounts.get_sketch_factory(
        num_buckets=4096),
    estimator=vector_of_counts.SequentialEstimator(),
    noiser=vector_of_counts.LaplaceNoiser(epsilon=np.log(3)))

VECTOR_OF_COUNTS_4096_0_SEQUENTIAL = SketchEstimatorConfig(
    name='vector_of_counts-4096-0-sequential',
    sketch_factory=vector_of_counts.VectorOfCounts.get_sketch_factory(
        num_buckets=4096),
    estimator=vector_of_counts.SequentialEstimator(),
    noiser=None)

SKETCH_ESTIMATOR_CONFIGS_TUPLE = (
    LOG_BLOOM_FILTER_1E5_LN3_FIRST_MOMENT_LOG,
    LOG_BLOOM_FILTER_1E5_0_FIRST_MOMENT_LOG,
    LIQUID_LEGIONS_1E5_10_LN3_SEQUENTIAL,
    LIQUID_LEGIONS_1E5_10_0_SEQUENTIAL,
    VECTOR_OF_COUNTS_4096_LN3_SEQUENTIAL,
    VECTOR_OF_COUNTS_4096_0_SEQUENTIAL)

NAME_TO_ESTIMATOR_CONFIGS = {
    conf.name: conf for conf in SKETCH_ESTIMATOR_CONFIGS_TUPLE}

ESTIMATOR_CONFIG_NAMES = tuple(NAME_TO_ESTIMATOR_CONFIGS.keys())
