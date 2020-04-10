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

The goal is to make sure that all of the estimators defined in estimators/ work
with the simulation and set generation code in simulations/
"""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BlipNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import SurrealDenoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UnionEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import CascadingLegions
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import Estimator
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import Noiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import AddRandomElementsNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HllCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HyperLogLogPlusPlus
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import LaplaceNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import SequentialEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import EstimatorConfig
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator


class InteroperabilityTest(absltest.TestCase):

  def setUp(self):
    super(InteroperabilityTest, self).setUp()
    self.number_of_trials = 2
    self.universe_size = 2000
    self.set_size = 5
    self.sketch_size = 64
    self.number_of_sets = 2
    self.order = set_generator.ORDER_RANDOM
    self.shared_prop = 0.2
    self.num_bloom_filter_hashes = 2
    self.bloom_filter_method = 'log'
    self.noiser_epsilon = np.log(3)
    self.noiser_flip_probability = .25

    self.set_random_state = np.random.RandomState(42)
    self.sketch_random_state = np.random.RandomState(137)
    self.noise_random_state = np.random.RandomState(3)

    # non-noised estimators
    estimator_config_cascading_legions = EstimatorConfig(
        sketch_factory=CascadingLegions.get_sketch_factory(
            self.sketch_size, self.sketch_size),
        estimator=Estimator(),
        noiser=None)

    estimator_config_bloom_filter = EstimatorConfig(
        sketch_factory=BloomFilter.get_sketch_factory(
            self.sketch_size, self.num_bloom_filter_hashes),
        estimator=UnionEstimator(),
        noiser=None)

    estimator_config_logarithmic_bloom_filter = EstimatorConfig(
        sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
            self.sketch_size),
        estimator=FirstMomentEstimator(method=self.bloom_filter_method),
        noiser=None)

    estimator_config_voc = EstimatorConfig(
        sketch_factory=VectorOfCounts.get_sketch_factory(self.sketch_size),
        estimator=SequentialEstimator(),
        noiser=None)

    estimator_config_exact = EstimatorConfig(
        sketch_factory=ExactSet.get_sketch_factory(),
        estimator=LosslessEstimator(),
        noiser=None)

    estimator_config_hll = EstimatorConfig(
        sketch_factory=HyperLogLogPlusPlus.get_sketch_factory(self.sketch_size),
        estimator=HllCardinality(),
        noiser=None)

    self.name_to_non_noised_estimator_config = {
        'exact_set': estimator_config_exact,
        'cascading_legions': estimator_config_cascading_legions,
        'bloom_filter': estimator_config_bloom_filter,
        'logarithmic_bloom_filter': estimator_config_logarithmic_bloom_filter,
        'vector_of_counts': estimator_config_voc,
        'hll': estimator_config_hll,
    }

    # noised estimators

    noised_estimator_config_cascading_legions = EstimatorConfig(
        sketch_factory=CascadingLegions.get_sketch_factory(
            self.sketch_size, self.sketch_size),
        estimator=Estimator(),
        noiser=Noiser(self.noiser_flip_probability))

    noised_estimator_config_bloom_filter = EstimatorConfig(
        sketch_factory=BloomFilter.get_sketch_factory(
            self.sketch_size, self.num_bloom_filter_hashes),
        estimator=UnionEstimator(),
        noiser=BlipNoiser(self.noiser_epsilon, self.noise_random_state))

    noised_estimator_config_logarithmic_bloom_filter = EstimatorConfig(
        sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
            self.sketch_size),
        estimator=FirstMomentEstimator(
            method=self.bloom_filter_method,
            denoiser=SurrealDenoiser(
                probability=self.noiser_flip_probability)),
        noiser=None)

    noised_estimator_config_voc = EstimatorConfig(
        sketch_factory=VectorOfCounts.get_sketch_factory(self.sketch_size),
        estimator=SequentialEstimator(),
        noiser=LaplaceNoiser())

    noised_estimator_config_exact = EstimatorConfig(
        sketch_factory=ExactSet.get_sketch_factory(),
        estimator=LosslessEstimator(),
        noiser=AddRandomElementsNoiser(1, self.noise_random_state))

    self.name_to_noised_estimator_config = {
        'exact_set': noised_estimator_config_exact,
        'cascading_legions': noised_estimator_config_cascading_legions,
        'bloom_filter': noised_estimator_config_bloom_filter,
        'logarithmic_bloom_filter':
            noised_estimator_config_logarithmic_bloom_filter,
        'vector_of_counts': noised_estimator_config_voc,
    }

  def simulate_with_set_generator(self, set_generator_factory, config_dict):
    for _, estimator_method_config in config_dict.items():

      simulator = Simulator(
          num_runs=self.number_of_trials,
          set_generator_factory=set_generator_factory,
          estimator_config=estimator_method_config,
          set_random_state=self.set_random_state,
          sketch_random_state=self.sketch_random_state)

      _, _ = simulator.run_all_and_aggregate()

  def test_with_independent_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.IndependentSetGenerator.get_generator_factory(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_independent_set_generator_noised(self):
    set_generator_factory = (
        set_generator.IndependentSetGenerator.get_generator_factory(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_fully_overlap_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.FullyOverlapSetGenerator.get_generator_factory(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_fully_overlap_set_generator_noised(self):
    set_generator_factory = (
        set_generator.FullyOverlapSetGenerator.get_generator_factory(
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)

  def test_with_sequentially_correlated_all_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.SequentiallyCorrelatedSetGenerator.get_generator_factory(
            order=self.order,
            correlated_sets=set_generator.CORRELATED_SETS_ALL,
            shared_prop=self.shared_prop,
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_non_noised_estimator_config)

  def test_with_sequentially_correlated_one_set_generator_non_noised(self):
    set_generator_factory = (
        set_generator.SequentiallyCorrelatedSetGenerator.get_generator_factory(
            order=self.order,
            correlated_sets=set_generator.CORRELATED_SETS_ONE,
            shared_prop=self.shared_prop,
            universe_size=self.universe_size,
            num_sets=self.number_of_sets,
            set_size=self.set_size))
    self.simulate_with_set_generator(set_generator_factory,
                                     self.name_to_noised_estimator_config)


if __name__ == '__main__':
  absltest.main()
