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
"""Generates example sets and estimates cardinality multiple ways, summarizes."""
from absl import app
from absl import flags
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import BloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UnionEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import CascadingLegions
from wfa_cardinality_estimation_evaluation_framework.estimators.cascading_legions import Estimator
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HllCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HyperLogLogPlusPlus
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import SequentialEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import EstimatorConfig
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator

FLAGS = flags.FLAGS

flags.DEFINE_integer('universe_size', 1000000,
                     'The number of unique possible user-ids')
flags.DEFINE_integer(
    'number_of_sets', 10,
    'The number of sets to depulicate across, AKA the number of publishers')
flags.DEFINE_integer('number_of_trials', 10,
                     'The number of times to run the experiment')
flags.DEFINE_integer('set_size', 1000, 'The size of all generated sets')
flags.DEFINE_integer('sketch_size', 8192, 'The size of sketches')
flags.DEFINE_integer('num_bloom_filter_hashes', 3,
                     'The number of hashes for the bloom filter to use')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  estimator_config_cascading_legions = EstimatorConfig(
      sketch_factory=CascadingLegions.get_sketch_factory(
          FLAGS.sketch_size, FLAGS.sketch_size),
      estimator=Estimator(),
      noiser=None)

  estimator_config_bloom_filter = EstimatorConfig(
      sketch_factory=BloomFilter.get_sketch_factory(
          FLAGS.sketch_size, FLAGS.num_bloom_filter_hashes),
      estimator=UnionEstimator(),
      noiser=None)

  estimator_config_logarithmic_bloom_filter = EstimatorConfig(
      sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
          FLAGS.sketch_size),
      estimator=FirstMomentEstimator(method='log'),
      noiser=None)

  estimator_config_voc = EstimatorConfig(
      sketch_factory=VectorOfCounts.get_sketch_factory(FLAGS.sketch_size),
      estimator=SequentialEstimator(),
      noiser=None)

  estimator_config_hll = EstimatorConfig(
      sketch_factory=HyperLogLogPlusPlus.get_sketch_factory(FLAGS.sketch_size),
      estimator=HllCardinality(),
      noiser=None)

  estimator_config_exact = EstimatorConfig(
      sketch_factory=ExactSet.get_sketch_factory(),
      estimator=LosslessEstimator(),
      noiser=None)

  name_to_estimator_config = {
      'bloom_filter': estimator_config_bloom_filter,
      'logarithmic_bloom_filter': estimator_config_logarithmic_bloom_filter,
      'cascading_legions': estimator_config_cascading_legions,
      'exact_set': estimator_config_exact,
      'hll++': estimator_config_hll,
      'vector_of_counts': estimator_config_voc,
  }
  set_generator_factory = (
      set_generator.IndependentSetGenerator.get_generator_factory(
          universe_size=FLAGS.universe_size,
          num_sets=FLAGS.number_of_sets,
          set_size=FLAGS.set_size))

  for method_name, estimator_method_config in name_to_estimator_config.items():
    print(f'Calculations for {method_name}')
    set_rs = np.random.RandomState(1)
    sketch_rs = np.random.RandomState(1)
    simulator = Simulator(
        num_runs=FLAGS.number_of_trials,
        set_generator_factory=set_generator_factory,
        estimator_config=estimator_method_config,
        set_random_state=set_rs,
        sketch_random_state=sketch_rs)

    _, agg_data = simulator.run_all_and_aggregate()
    print(f'Aggregate Statistics for {method_name}')
    print(agg_data)


if __name__ == '__main__':
  app.run(main)
