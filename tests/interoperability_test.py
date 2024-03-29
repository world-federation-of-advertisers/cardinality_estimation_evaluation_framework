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
"""Tests for wfa_cardinality_estimation_evaluation_framework.

The goals are to make sure that (1) all of the estimators defined in estimators/
work with the simulation and set generation code in simulations/, and (2) the
estimator, set generator and simulator works with the evaluator, analyzer and
report generator.
"""
import math
from absl.testing import absltest
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BlipNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import ExponentialBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import GeometricBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import SurrealDenoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UnionEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import CascadingLegions
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import Estimator
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import Noiser
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import GeometricEstimateNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import AddRandomElementsNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import FreqLogLogCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.freq_log_log import FreqLogLogPlusPlus
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HllCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HyperLogLogPlusPlus
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import LaplaceNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import SequentialEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts
from wfa_cardinality_estimation_evaluation_framework.evaluations import run_evaluation
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import SketchEstimatorConfig
from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator


class InteroperabilityTest(absltest.TestCase):

  def setUp(self):
    super(InteroperabilityTest, self).setUp()
    self.number_of_trials = 2
    self.universe_size = 2000
    self.set_size_list = [5, 7, 9]
    self.large_set_size = 6
    self.small_set_size = 3
    self.sketch_size = 128
    self.number_of_sets = 3
    self.set_size = 50
    self.num_large_sets = 1
    self.num_small_sets = 3
    self.order = set_generator.ORDER_RANDOM
    self.user_activity_association = (
        set_generator.USER_ACTIVITY_ASSOCIATION_INDEPENDENT)
    self.shared_prop = 0.2
    self.num_bloom_filter_hashes = 2
    self.exponential_bloom_filter_decay_rate = 10
    self.geometic_bloom_filter_probability = 0.08
    self.noiser_epsilon = np.log(3)
    self.noiser_flip_probability = .25

    self.set_random_state = np.random.RandomState(42)
    self.sketch_random_state = np.random.RandomState(137)
    self.noise_random_state = np.random.RandomState(3)

    # non-noised estimators
    estimator_config_cascading_legions = SketchEstimatorConfig(
        name='cascading_legions',
        sketch_factory=CascadingLegions.get_sketch_factory(
            self.sketch_size, self.sketch_size),
        estimator=Estimator())

    estimator_config_bloom_filter = SketchEstimatorConfig(
        name='bloom_filter-union_estimator',
        sketch_factory=BloomFilter.get_sketch_factory(
            self.sketch_size, self.num_bloom_filter_hashes),
        estimator=UnionEstimator())

    estimator_config_geometric_bloom_filter = SketchEstimatorConfig(
        name='geo_bloom_filter-first_moment_geo',
        sketch_factory=GeometricBloomFilter.get_sketch_factory(
            self.sketch_size, self.geometic_bloom_filter_probability),
        estimator=FirstMomentEstimator(method='geo'))

    estimator_config_logarithmic_bloom_filter = SketchEstimatorConfig(
        name='log_bloom_filter-first_moment_log',
        sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
            self.sketch_size),
        estimator=FirstMomentEstimator(method='log'))

    estimator_config_exponential_bloom_filter = SketchEstimatorConfig(
        name='exp_bloom_filter-first_moment_exp',
        sketch_factory=ExponentialBloomFilter.get_sketch_factory(
            self.sketch_size, self.exponential_bloom_filter_decay_rate),
        estimator=FirstMomentEstimator(method='exp'))

    estimator_config_voc = SketchEstimatorConfig(
        name='vector_of_counts-sequential',
        sketch_factory=VectorOfCounts.get_sketch_factory(self.sketch_size),
        estimator=SequentialEstimator())

    estimator_config_exact = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=ExactMultiSet.get_sketch_factory(),
        estimator=LosslessEstimator())

    estimator_config_fll = SketchEstimatorConfig(
        name='freq_log_log',
        sketch_factory=FreqLogLogPlusPlus.get_sketch_factory(self.sketch_size),
        estimator=FreqLogLogCardinality())

    estimator_config_hll = SketchEstimatorConfig(
        name='hyper_log_log',
        sketch_factory=HyperLogLogPlusPlus.get_sketch_factory(self.sketch_size),
        estimator=HllCardinality())

    estimator_config_expadbf_first_moment_global_dp = SketchEstimatorConfig(
        name='estimator_config_expadbf_first_moment_global_d',
        sketch_factory=ExponentialBloomFilter.get_sketch_factory(
            length=10**5, decay_rate=10),
        estimator=FirstMomentEstimator(
            method=FirstMomentEstimator.METHOD_EXP,
            noiser=GeometricEstimateNoiser(epsilon=math.log(3))))

    config_list = [
        estimator_config_exact,
        estimator_config_cascading_legions,
        estimator_config_bloom_filter,
        estimator_config_logarithmic_bloom_filter,
        estimator_config_exponential_bloom_filter,
        estimator_config_geometric_bloom_filter,
        estimator_config_voc,
        estimator_config_fll,
        estimator_config_hll,
        estimator_config_expadbf_first_moment_global_dp,
    ]

    self.name_to_non_noised_estimator_config = {
        config.name: config for config in config_list
    }

    # noised estimators
    noised_estimator_config_cascading_legions = SketchEstimatorConfig(
        name='cascading_legions',
        sketch_factory=CascadingLegions.get_sketch_factory(
            self.sketch_size, self.sketch_size),
        estimator=Estimator(),
        sketch_noiser=Noiser(self.noiser_flip_probability))

    noised_estimator_config_bloom_filter = SketchEstimatorConfig(
        name='bloom_filter-union_estimator',
        sketch_factory=BloomFilter.get_sketch_factory(
            self.sketch_size, self.num_bloom_filter_hashes),
        estimator=UnionEstimator(),
        sketch_noiser=BlipNoiser(self.noiser_epsilon, self.noise_random_state))

    noised_estimator_config_geometric_bloom_filter = SketchEstimatorConfig(
        name='geo_bloom_filter-first_moment_geo',
        sketch_factory=GeometricBloomFilter.get_sketch_factory(
            self.sketch_size, self.geometic_bloom_filter_probability),
        estimator=FirstMomentEstimator(
            method='geo',
            denoiser=SurrealDenoiser(epsilon=math.log(3))),
        sketch_noiser=BlipNoiser(self.noiser_epsilon, self.noise_random_state))

    noised_estimator_config_logarithmic_bloom_filter = SketchEstimatorConfig(
        name='log_bloom_filter-first_moment_log',
        sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
            self.sketch_size),
        estimator=FirstMomentEstimator(
            method='log',
            denoiser=SurrealDenoiser(epsilon=math.log(3))),
        sketch_noiser=BlipNoiser(self.noiser_epsilon, self.noise_random_state))

    noised_estimator_config_exponential_bloom_filter = SketchEstimatorConfig(
        name='exp_bloom_filter-first_moment_exp',
        sketch_factory=ExponentialBloomFilter.get_sketch_factory(
            self.sketch_size, self.exponential_bloom_filter_decay_rate),
        estimator=FirstMomentEstimator(
            method='exp',
            denoiser=SurrealDenoiser(epsilon=math.log(3))),
        sketch_noiser=BlipNoiser(self.noiser_epsilon, self.noise_random_state))

    noised_estimator_config_voc = SketchEstimatorConfig(
        name='vector_of_counts-sequential',
        sketch_factory=VectorOfCounts.get_sketch_factory(self.sketch_size),
        estimator=SequentialEstimator(),
        sketch_noiser=LaplaceNoiser())

    noised_estimator_config_exact = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=ExactMultiSet.get_sketch_factory(),
        estimator=LosslessEstimator(),
        sketch_noiser=AddRandomElementsNoiser(1, self.noise_random_state))

    noised_config_list = [
        noised_estimator_config_exact,
        noised_estimator_config_cascading_legions,
        noised_estimator_config_bloom_filter,
        noised_estimator_config_logarithmic_bloom_filter,
        noised_estimator_config_exponential_bloom_filter,
        noised_estimator_config_geometric_bloom_filter,
        noised_estimator_config_voc,
    ]

    self.name_to_noised_estimator_config = {
        config.name: config for config in noised_config_list
    }

  def simulate_with_set_generator(self, set_generator_factory, config_dict):
    for _, estimator_method_config in config_dict.items():

      simulator = Simulator(
          num_runs=self.number_of_trials,
          set_generator_factory=set_generator_factory,
          sketch_estimator_config=estimator_method_config,
          set_random_state=self.set_random_state,
          sketch_random_state=self.sketch_random_state)

      _, _ = simulator.run_all_and_aggregate()

  def test_with_independent_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_independent_set_generator_non_noised_different_sizes(self):
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_set_size_list(
            universe_size=self.universe_size,
            set_size_list=self.set_size_list))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_independent_set_generator_noised(self):
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_independent_set_generator_noised_different_sizes(self):
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_set_size_list(
            universe_size=self.universe_size,
            set_size_list=self.set_size_list))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_exponential_bow_set_generator_non_noised(self):
    # Choose special sizes here because Exponential Bow requires minimum size
    # See set_generator.ExponentialBowSetGenerator for details
    set_generator_factory = (
        set_generator.ExponentialBowSetGenerator.
        get_generator_factory_with_num_and_size(
            user_activity_association=self.user_activity_association,
            universe_size=200, num_sets=2, set_size=50))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_exponential_bow_set_generator_non_noised_different_sizes(self):
    # Choose special sizes here because Exponential Bow requires minimum size
    # See set_generator.ExponentialBowSetGenerator for details
    set_generator_factory = (
        set_generator.ExponentialBowSetGenerator.
        get_generator_factory_with_set_size_list(
            user_activity_association=self.user_activity_association,
            universe_size=200, set_size_list=[50, 60, 70]))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_exponential_bow_set_generator_noised(self):
    set_generator_factory = (
        set_generator.ExponentialBowSetGenerator.
        get_generator_factory_with_num_and_size(
            user_activity_association=self.user_activity_association,
            universe_size=200, num_sets=2, set_size=50))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_exponential_bow_set_generator_noised_different_sizes(self):
    set_generator_factory = (
        set_generator.ExponentialBowSetGenerator.
        get_generator_factory_with_set_size_list(
            user_activity_association=self.user_activity_association,
            universe_size=200, set_size_list=[50, 60, 70]))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_fully_overlap_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.FullyOverlapSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_fully_overlap_set_generator_noised(self):
    set_generator_factory = (
        set_generator.FullyOverlapSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_sub_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.SubSetGenerator.get_generator_factory_with_num_and_size(
            order=self.order,
            universe_size=self.universe_size,
            num_large_sets=self.num_large_sets,
            num_small_sets=self.num_small_sets,
            large_set_size=self.large_set_size,
            small_set_size=self.small_set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_sub_set_generator_noised(self):
    set_generator_factory = (
        set_generator.SubSetGenerator.get_generator_factory_with_num_and_size(
            order=self.order,
            universe_size=self.universe_size,
            num_large_sets=self.num_large_sets,
            num_small_sets=self.num_small_sets,
            large_set_size=self.large_set_size,
            small_set_size=self.small_set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_sequentially_correlated_all_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.SequentiallyCorrelatedSetGenerator.
        get_generator_factory_with_num_and_size(
            order=self.order,
            correlated_sets=set_generator.CORRELATED_SETS_ALL,
            shared_prop=self.shared_prop,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  # Using abbreviation seq_corr here, otherwise the function name is too long.
  def test_with_seq_corr_all_set_generator_non_noised_different_sizes(self):
    set_generator_factory = (
        set_generator.SequentiallyCorrelatedSetGenerator.
        get_generator_factory_with_set_size_list(
            order=self.order,
            correlated_sets=set_generator.CORRELATED_SETS_ALL,
            shared_prop=self.shared_prop,
            set_size_list=self.set_size_list))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_sequentially_correlated_one_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.SequentiallyCorrelatedSetGenerator.
        get_generator_factory_with_num_and_size(
            order=self.order,
            correlated_sets=set_generator.CORRELATED_SETS_ONE,
            shared_prop=self.shared_prop,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_seq_corr_one_set_generator_non_noised_different_sizes(self):
    set_generator_factory = (
        set_generator.SequentiallyCorrelatedSetGenerator.
        get_generator_factory_with_set_size_list(
            order=self.order,
            correlated_sets=set_generator.CORRELATED_SETS_ONE,
            shared_prop=self.shared_prop,
            set_size_list=self.set_size_list))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_run_evaluation_for_cardinality_estimator_workflow_compatible(self):
    """Test the compatibility of evaluator, analyzer and report_generator.

    This is a test to check if the evaluator, the analyzer and the
    report_generator is compatible with the rest of the evaluation codebase,
    eg, the estimators, the set generators, and the simulator. The test runs
    the evaluation, analyzes results, and generates a report, which should not
    run into any error.
    """
    sketch_estimator_configs = [conf.name for conf in (
        evaluation_configs._generate_cardinality_estimator_configs())]
    run_evaluation._run(
        run_evaluation=True,
        run_analysis=True,
        generate_html_report=True,
        evaluation_out_dir=self.create_tempdir('evaluator').full_path,
        analysis_out_dir=self.create_tempdir('analyzer').full_path,
        report_out_dir=self.create_tempdir('report').full_path,
        evaluation_config='smoke_test',
        sketch_estimator_configs=sketch_estimator_configs,
        evaluation_run_name='interoperability_test_for_evaluator_cardinality',
        num_runs=1,
        universe_size=1000,
        num_workers=0,
        error_margin=[0.05],
        proportion_of_runs=[0.95],
        boxplot_xlabel_rotate=90,
        boxplot_size_width_inch=6,
        boxplot_size_height_inch=4,
        analysis_type='cardinality',
        max_frequency=10
    )

  def test_run_evaluation_for_frequency_estimator_workflow_compatible(self):
    """Test the compatibility of evaluator, analyzer and report_generator.

    This is a test to check if the evaluator, the analyzer and the
    report_generator is compatible with the rest of the evaluation codebase,
    eg, the estimators, the set generators, and the simulator. The test runs
    the evaluation, analyzes results, and generates a report, which should not
    run into any error.
    """
    max_frequency = 3
    sketch_estimator_configs = [conf.name for conf in (
        evaluation_configs._generate_frequency_estimator_configs(max_frequency)
    )]
    run_evaluation._run(
        run_evaluation=True,
        run_analysis=True,
        generate_html_report=True,
        evaluation_out_dir=self.create_tempdir('evaluator').full_path,
        analysis_out_dir=self.create_tempdir('analyzer').full_path,
        report_out_dir=self.create_tempdir('report').full_path,
        evaluation_config='frequency_end_to_end_test',
        sketch_estimator_configs=sketch_estimator_configs,
        evaluation_run_name='interoperability_test_for_evaluator_frequency',
        num_runs=1,
        universe_size=1000,
        num_workers=0,
        error_margin=[0.05],
        proportion_of_runs=[0.95],
        boxplot_xlabel_rotate=90,
        boxplot_size_width_inch=6,
        boxplot_size_height_inch=4,
        barplot_size_width_inch=6,
        barplot_size_height_inch=4,
        analysis_type='frequency',
        max_frequency=max_frequency,
    )

if __name__ == '__main__':
  absltest.main()
