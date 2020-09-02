# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate the influence of set size / sketch size."""

from absl import app
from absl import flags
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
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import AddRandomElementsNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HllCardinality
from wfa_cardinality_estimation_evaluation_framework.estimators.hyper_log_log import HyperLogLogPlusPlus
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import LaplaceNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import SequentialEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import VectorOfCounts
from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import configs
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.evaluations import report_generator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator
from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import SketchEstimatorConfig

FLAGS = flags.FLAGS

flags.DEFINE_integer('universe_size', 1000000,
                     'The number of unique possible user-ids')
flags.DEFINE_integer(
    'number_of_sets', 3,
    'The number of sets to depulicate across, AKA the number of publishers')
flags.DEFINE_integer('number_of_trials', 50,
                     'The number of times to run the experiment')
flags.DEFINE_list(
    'set_size_ratio', 
    [.5, 1, 2, 4, 8, 16, 32, 64], 
    'The size of all generated sets')
flags.DEFINE_integer('sketch_size', 10000, 'The size of sketches')
flags.DEFINE_integer('exponential_bloom_filter_decay_rate', 10,
                     'The decay rate in exponential bloom filter')
flags.DEFINE_integer('num_bloom_filter_hashes', 3,
                     'The number of hashes for the bloom filter to use')
flags.DEFINE_float('geometric_bloom_filter_probability', 0.0015,
                    'probability of geometric distribution')
flags.DEFINE_float("noiser_epsilon", np.sqrt(3), 
                   "target privacy parameter in noiser")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    noiser_flip_probability = 1 / (1 + np.exp(FLAGS.noiser_epsilon))

    ## list three adbf estimators
    estimator_config_geometric_bloom_filter = SketchEstimatorConfig(
        name='geo_BF',
        sketch_factory=GeometricBloomFilter.get_sketch_factory(
            FLAGS.sketch_size, FLAGS.geometric_bloom_filter_probability),
        estimator=FirstMomentEstimator(
            method='geo',
            denoiser=SurrealDenoiser(
                probability=noiser_flip_probability)), 
        sketch_noiser=BlipNoiser(FLAGS.noiser_epsilon))

    estimator_config_logarithmic_bloom_filter = SketchEstimatorConfig(
        name='log_BF',
        sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
            FLAGS.sketch_size),
        estimator=FirstMomentEstimator(
            method='log',
            denoiser=SurrealDenoiser(
                probability=noiser_flip_probability)), 
        sketch_noiser=BlipNoiser(FLAGS.noiser_epsilon))

    estimator_config_exponential_bloom_filter = SketchEstimatorConfig(
        name='exp_BF',
        sketch_factory=ExponentialBloomFilter.get_sketch_factory(
            FLAGS.sketch_size, FLAGS.exponential_bloom_filter_decay_rate),
        estimator=FirstMomentEstimator(
            method='exp',
            denoiser=SurrealDenoiser(
                probability=noiser_flip_probability)), 
        sketch_noiser=BlipNoiser(FLAGS.noiser_epsilon))
        
    estimator_config_list = [
        estimator_config_geometric_bloom_filter,
        estimator_config_logarithmic_bloom_filter,
        estimator_config_exponential_bloom_filter,
    ]

    # list scenarios of different set sizes
    scenario_config_list = []
    for set_size_ratio in FLAGS.set_size_ratio: 
        set_size = int(set_size_ratio * FLAGS.sketch_size)
        scenario_config_list += [
            configs.ScenarioConfig(
                name="{:.1f}".format(set_size_ratio),
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory_with_num_and_size(
                        universe_size=FLAGS.universe_size, 
                        num_sets=FLAGS.number_of_sets, 
                        set_size=set_size)))
        ]
    evaluation_config = configs.EvaluationConfig(
        name='2_vary_set_size',
        num_runs=FLAGS.number_of_trials,
        scenario_config_list=scenario_config_list)

    generate_results = evaluator.Evaluator(
        evaluation_config=evaluation_config,
        sketch_estimator_config_list=estimator_config_list,
        run_name="eval_adbf_result",
        out_dir=".",
        workers=10)
    generate_results()

if __name__ == '__main__':
  app.run(main)
