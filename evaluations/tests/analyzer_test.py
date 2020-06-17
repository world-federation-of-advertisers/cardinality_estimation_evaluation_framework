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

"""Tests for wfa_cardinality_estimation_evaluation_framework.evaluations.analyzer."""
import itertools
import os

from absl.testing import absltest

import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.estimators import exact_set
from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import configs
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


class AnalyzerTest(absltest.TestCase):

  def setUp(self):
    super(AnalyzerTest, self).setUp()
    exact_set_lossless = configs.SketchEstimatorConfig(
        name='exact_set_lossless',
        sketch_factory=exact_set.ExactSet.get_sketch_factory(),
        estimator=exact_set.LosslessEstimator(),
        noiser=None)
    exact_set_less_one = configs.SketchEstimatorConfig(
        name='exact_set_less_one',
        sketch_factory=exact_set.ExactSet.get_sketch_factory(),
        estimator=exact_set.LessOneEstimator(),
        noiser=exact_set.AddRandomElementsNoiser(
            num_random_elements=0, random_state=np.random.RandomState()))
    self.sketch_estimator_config_list = (exact_set_lossless, exact_set_less_one)

    self.evaluation_config = configs.EvaluationConfig(
        name='test_evaluation',
        num_runs=2,
        scenario_config_list=[
            configs.ScenarioConfig(
                name='ind1',
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory(
                        universe_size=10, num_sets=5, set_size=1))),
            configs.ScenarioConfig(
                name='ind2',
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory(
                        universe_size=10, num_sets=5, set_size=1))),
        ])

    self.run_name = 'test_run'

    def _get_test_evaluator(out_dir):
      return evaluator.Evaluator(
          evaluation_config=self.evaluation_config,
          sketch_estimator_config_list=self.sketch_estimator_config_list,
          run_name=self.run_name,
          out_dir=out_dir)

    self.get_test_evaluator = _get_test_evaluator

    def _get_test_analyzer(out_dir, evaluation_dir):
      return analyzer.CardinalityEstimatorEvaluationAnalyzer(
          out_dir=out_dir,
          evaluation_directory=evaluation_dir,
          evaluation_run_name=self.run_name,
          evaluation_name=self.evaluation_config.name)

    self.get_test_analyzer = _get_test_analyzer

  def test_get_num_estimable_sets_function_works(self):
    df = pd.DataFrame({
        'run': list(itertools.chain(*[range(4)] * 4)),
        'number_of_sets': [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
        'error': [0] * 4 + [0, 0, 1, -1] + [1, 1, -1, -1] + [0.5, -0.5, -1, 1],
    })
    result = analyzer.get_num_estimable_sets(
        df, num_sets='number_of_sets', relative_error='error',
        error_margin=0.5, proportion_of_runs=0.5)
    expected = 2
    self.assertEqual(result, expected)

  def test_read_evaluation_results_works(self):
    # Run evaluation.
    evaluation_out_dir = self.create_tempdir('evaluation')
    e = self.get_test_evaluator(out_dir=evaluation_out_dir)
    e()
    # Get results.
    a = self.get_test_analyzer(
        out_dir=self.create_tempdir('analysis'),
        evaluation_dir=evaluation_out_dir)
    raw_results = a.raw_df

    estimators = ('exact_set_lossless', 'exact_set_less_one')
    scenarios = ('ind1', 'ind2')
    for _, row in raw_results.iterrows():
      self.assertIn(row[analyzer.ESTIMATOR_NAME], estimators)
      self.assertIn(row[analyzer.SCENARIO_NAME], scenarios)
      self.assertIsInstance(
          row[analyzer.RAW_RESULT_DF], pd.core.frame.DataFrame,
          'The result is not a DataFrame.')

  def test_get_num_estimable_sets_df_results_correct(self):
    evaluation_out_dir = self.create_tempdir('evaluation')
    # Run evaluation.
    e = self.get_test_evaluator(evaluation_out_dir)
    e()
    # Get analyzer.
    analysis_out_dir = self.create_tempdir('analysis')
    a = self.get_test_analyzer(
        out_dir=analysis_out_dir,
        evaluation_dir=evaluation_out_dir)
    df = a.get_num_estimable_sets_df().set_index([
        analyzer.ESTIMATOR_NAME, analyzer.SCENARIO_NAME]).sort_index()
    expected = pd.DataFrame({
        analyzer.ESTIMATOR_NAME: (
            ['exact_set_lossless'] * 2 + ['exact_set_less_one'] * 2),
        analyzer.SCENARIO_NAME: ['ind1', 'ind2'] * 2,
        analyzer.NUM_ESTIMABLE_SETS: [5, 5, 0, 0]
    }).set_index([analyzer.ESTIMATOR_NAME, analyzer.SCENARIO_NAME]).sort_index()
    try:
      pd.testing.assert_frame_equal(df, expected)
    except AssertionError:
      self.fail('The number of estimable sets is not correct.')

  def test_save_plot_num_sets_vs_relative_error(self):
    evaluation_out_dir = self.create_tempdir('evaluation')
    # Run evaluation.
    e = self.get_test_evaluator(evaluation_out_dir)
    e()
    # Get analyzer.
    a = self.get_test_analyzer(
        out_dir=self.create_tempdir('analysis'),
        evaluation_dir=evaluation_out_dir)
    a.save_plot_num_sets_vs_relative_error()
    print(a.analysis_file_dirs[evaluator.KEY_ESTIMATOR_DIRS])
    for estimator in a.analysis_file_dirs[evaluator.KEY_ESTIMATOR_DIRS].keys():
      for scenario, directory in a.analysis_file_dirs[estimator].items():
        self.assertTrue(
            os.path.exists(os.path.join(directory, analyzer.BOXPLOT_FILENAME)),
            f'Plot file does not exists for {estimator} under {scenario}.')


if __name__ == '__main__':
  absltest.main()
