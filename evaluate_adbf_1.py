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

"""Evaluate the influence of flipping probability."""

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
    'number_of_sets', 10,
    'The number of sets to depulicate across, AKA the number of publishers')
flags.DEFINE_integer('number_of_trials', 100,
                     'The number of times to run the experiment')
flags.DEFINE_integer('set_size', 50000, 'The size of all generated sets')
flags.DEFINE_integer('sketch_size', 10000, 'The size of sketches')
flags.DEFINE_integer('exponential_bloom_filter_decay_rate', 10,
                     'The decay rate in exponential bloom filter')
flags.DEFINE_integer('num_bloom_filter_hashes', 3,
                     'The number of hashes for the bloom filter to use')
flags.DEFINE_float('geometric_bloom_filter_probability', 0.0015,
                    'probability of geometric distribution')
flags.DEFINE_list("noiser_epsilon", 
                  [np.log(9), np.sqrt(3), np.log(4), np.log(3)],
                  "target privacy parameter in noiser")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    ## list all estimators
    estimator_config_list = []
    for epsilon in FLAGS.noiser_epsilon:
        ## flipping prob
        noiser_flip_probability = 1 / (1 + np.exp(epsilon))

        # estimator_config_bloom_filter = SketchEstimatorConfig(
        #     name='unif_BF_' + "{:.2f}".format(noiser_flip_probability),
        #     sketch_factory=BloomFilter.get_sketch_factory(
        #         FLAGS.sketch_size, FLAGS.num_bloom_filter_hashes),
        #     estimator=UnionEstimator(),
        #     sketch_noiser=BlipNoiser(epsilon))

        estimator_config_geometric_bloom_filter = SketchEstimatorConfig(
            name='geo_BF_' + "{:.2f}".format(noiser_flip_probability),
            sketch_factory=GeometricBloomFilter.get_sketch_factory(
                FLAGS.sketch_size, FLAGS.geometric_bloom_filter_probability),
            estimator=FirstMomentEstimator(
                method='geo',
                denoiser=SurrealDenoiser(epsilon)), 
            sketch_noiser=BlipNoiser(epsilon))

        estimator_config_logarithmic_bloom_filter = SketchEstimatorConfig(
            name='log_BF_' + "{:.2f}".format(noiser_flip_probability),
            sketch_factory=LogarithmicBloomFilter.get_sketch_factory(
                FLAGS.sketch_size),
            estimator=FirstMomentEstimator(
                method='log',
                denoiser=SurrealDenoiser(epsilon)), 
            sketch_noiser=BlipNoiser(epsilon))

        estimator_config_exponential_bloom_filter = SketchEstimatorConfig(
            name='exp_BF_' + "{:.2f}".format(noiser_flip_probability),
            sketch_factory=ExponentialBloomFilter.get_sketch_factory(
                FLAGS.sketch_size, FLAGS.exponential_bloom_filter_decay_rate),
            estimator=FirstMomentEstimator(
                method='exp',
                denoiser=SurrealDenoiser(epsilon)), 
            sketch_noiser=BlipNoiser(epsilon))
    
        estimator_config_list += [
            # estimator_config_bloom_filter,
            estimator_config_geometric_bloom_filter,
            estimator_config_logarithmic_bloom_filter,
            estimator_config_exponential_bloom_filter,
        ]

    # config evaluation 
    evaluation_config = configs.EvaluationConfig(
        name='1_vary_flip_prob',
        num_runs=FLAGS.number_of_trials,
        scenario_config_list=[
            configs.ScenarioConfig(
                name='independent',
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory_with_num_and_size(
                        universe_size=FLAGS.universe_size, 
                        num_sets=FLAGS.number_of_sets, 
                        set_size=FLAGS.set_size)))
        ])

    generate_results = evaluator.Evaluator(
        evaluation_config=evaluation_config,
        sketch_estimator_config_list=estimator_config_list,
        run_name="eval_adbf_result",
        out_dir=".",
        workers=10)
    generate_results()

    # set_generator_factory = (
    #     set_generator.IndependentSetGenerator.
    #     get_generator_factory_with_num_and_size(
    #         universe_size=FLAGS.universe_size,
    #         num_sets=FLAGS.number_of_sets,
    #         set_size=FLAGS.set_size))

    # for estimator_method_config in estimator_config_list:
    #     print(f'Calculations for {estimator_method_config.name}')
    #     # set_rs = np.random.RandomState(1)
    #     # sketch_rs = np.random.RandomState(1)
    #     simulator = Simulator(
    #         num_runs=FLAGS.number_of_trials,
    #         set_generator_factory=set_generator_factory,
    #         sketch_estimator_config=estimator_method_config,
    #         # set_random_state=set_rs,
    #         # sketch_random_state=sketch_rs
    #     )

    #     raw_data, agg_data = simulator.run_all_and_aggregate()
    #     print(f'Aggregate Statistics for {estimator_method_config.name}')
    #     row_data['flip_prob'] = noiser_flip_probability
    #     data = data.append(raw_data)
    #     print(agg_data)

    # data.to_csv("data1.csv")


if __name__ == '__main__':
  app.run(main)
