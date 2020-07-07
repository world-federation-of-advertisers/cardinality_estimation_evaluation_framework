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

EVALUATION_CONFIGS_TUPLE = (
    _smoke_test,
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
