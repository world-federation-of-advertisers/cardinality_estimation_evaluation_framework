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
import collections
import itertools
import os

from absl.testing import absltest

import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.estimators import exact_set
from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import configs
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs
from wfa_cardinality_estimation_evaluation_framework.simulations import frequency_set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator


class AnalyzerTest(absltest.TestCase):

  def setUp(self):
    super(AnalyzerTest, self).setUp()
    exact_set_lossless = configs.SketchEstimatorConfig(
        name='exact_set-infty-infty-lossless',
        sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
        estimator=exact_set.LosslessEstimator())
    exact_set_less_one = configs.SketchEstimatorConfig(
        name='exact_set-infty-infty-less_one',
        sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
        estimator=exact_set.LessOneEstimator(),
        sketch_noiser=exact_set.AddRandomElementsNoiser(
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
                    .get_generator_factory_with_num_and_size(
                        universe_size=10, num_sets=5, set_size=1))),
            configs.ScenarioConfig(
                name='ind2',
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory_with_num_and_size(
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
          evaluation_name=self.evaluation_config.name,
          estimable_criteria_list=[(0.05, 0.95), (1.01, 0.9)])

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

  def test_get_num_estimable_sets_function_ignore_one_set(self):
    df = pd.DataFrame({
        'run': list(itertools.chain(*[range(4)] * 4)),
        'number_of_sets': [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
        'error': [1] * 4 + [0, 0, 1, -1] + [1, 1, -1, -1] + [0.5, -0.5, -1, 1],
    })
    result = analyzer.get_num_estimable_sets(
        df, num_sets='number_of_sets', relative_error='error',
        error_margin=0.5, proportion_of_runs=0.5)
    expected = 2
    self.assertEqual(result, expected)

  def test_read_evaluation_results_works(self):
    # Run evaluation.
    evaluation_out_dir = self.create_tempdir('evaluation').full_path
    e = self.get_test_evaluator(out_dir=evaluation_out_dir)
    e()
    # Get results.
    a = self.get_test_analyzer(
        out_dir=self.create_tempdir('analysis').full_path,
        evaluation_dir=evaluation_out_dir)
    raw_results = a.raw_df

    estimators = ('exact_set-infty-infty-lossless',
                  'exact_set-infty-infty-less_one')
    scenarios = ('ind1', 'ind2')
    for _, row in raw_results.iterrows():
      self.assertIn(row[analyzer.SKETCH_ESTIMATOR_NAME], estimators)
      self.assertIn(row[analyzer.SCENARIO_NAME], scenarios)
    self.assertEqual(raw_results.shape[0], 40)

  def test_get_num_estimable_sets_df_results_correct(self):
    evaluation_out_dir = self.create_tempdir('evaluation').full_path
    # Run evaluation.
    e = self.get_test_evaluator(evaluation_out_dir)
    e()
    # Get analyzer.
    analysis_out_dir = self.create_tempdir('analysis').full_path
    a = self.get_test_analyzer(
        out_dir=analysis_out_dir,
        evaluation_dir=evaluation_out_dir)
    df = a.get_num_estimable_sets_df()
    cols = [
        analyzer.ERROR_MARGIN_NAME, analyzer.PROPORTION_OF_RUNS_NAME,
        analyzer.SKETCH_ESTIMATOR_NAME, analyzer.SCENARIO_NAME]
    df = df.set_index(cols).sort_index()
    expected = pd.DataFrame({
        analyzer.ERROR_MARGIN_NAME: [0.05] * 4 + [1.01] * 4,
        analyzer.PROPORTION_OF_RUNS_NAME: [0.95] * 4 + [0.9] * 4,
        analyzer.SKETCH_ESTIMATOR_NAME: [
            'exact_set-infty-infty-lossless', 'exact_set-infty-infty-lossless',
            'exact_set-infty-infty-less_one', 'exact_set-infty-infty-less_one'
        ] * 2,
        analyzer.SCENARIO_NAME: ['ind1', 'ind2'] * 4,
        analyzer.NUM_ESTIMABLE_SETS: [5, 5, 0, 0, 5, 5, 5, 5]
    })
    expected = expected.set_index(cols).sort_index()
    try:
      pd.testing.assert_frame_equal(df, expected)
    except AssertionError:
      self.fail('The number of estimable sets is not correct.')

  def test_get_relative_error_stats_of_num_of_estimable_sets(self):
    evaluation_out_dir = self.create_tempdir('evaluation').full_path
    # Run evaluation.
    e = self.get_test_evaluator(evaluation_out_dir)
    e()
    # Get analyzer.
    analysis_out_dir = self.create_tempdir('analysis').full_path
    a = self.get_test_analyzer(
        out_dir=analysis_out_dir,
        evaluation_dir=evaluation_out_dir)
    df = a.get_relative_error_stats_of_num_of_estimable_sets()
    self.assertEqual(df.shape[0], 8, 'Missing rows.')
    self.assertSameElements(
        df.columns,
        [analyzer.ERROR_MARGIN_NAME, analyzer.PROPORTION_OF_RUNS_NAME,
         analyzer.SKETCH_ESTIMATOR_NAME, analyzer.SCENARIO_NAME,
         analyzer.NUM_ESTIMABLE_SETS,
         simulator.RELATIVE_ERROR_BASENAME + '1_mean',
         simulator.RELATIVE_ERROR_BASENAME + '1_std'],
        'num_of_estimable_sets missing columns.')
    df_lossless = df.loc[
        df[analyzer.SKETCH_ESTIMATOR_NAME] == 'exact_set-infty-infty-lossless']
    self.assertTrue(
        all(df_lossless[simulator.RELATIVE_ERROR_BASENAME + '1_mean'] == 0),
        'Relative error is not correct.')

  def test_save_plot_num_sets_vs_relative_error(self):
    evaluation_out_dir = self.create_tempdir('evaluation').full_path
    # Run evaluation.
    e = self.get_test_evaluator(evaluation_out_dir)
    e()
    # Get analyzer.
    a = self.get_test_analyzer(
        out_dir=self.create_tempdir('analysis').full_path,
        evaluation_dir=evaluation_out_dir)
    a._save_plot_num_sets_vs_metric()
    for estimator in a.analysis_file_dirs[evaluator.KEY_ESTIMATOR_DIRS].keys():
      for scenario, directory in a.analysis_file_dirs[estimator].items():
        self.assertTrue(
            os.path.exists(os.path.join(directory, analyzer.BOXPLOT_FILENAME)),
            f'Plot file does not exists for {estimator} under {scenario}.')

  def test_get_analysis_results(self):
    evaluation_out_dir = self.create_tempdir('evaluation').full_path
    # Run evaluation.
    e = self.get_test_evaluator(evaluation_out_dir)
    e()
    # Get analyzer.
    analysis_dir = self.create_tempdir('analysis').full_path
    a = self.get_test_analyzer(
        out_dir=analysis_dir,
        evaluation_dir=evaluation_out_dir)
    a()
    # Get analysis results.
    analysis_results = analyzer.get_analysis_results(
        analysis_out_dir=analysis_dir,
        evaluation_run_name=self.run_name,
        evaluation_name=self.evaluation_config.name)

    # Check if the results contain the correct members.
    self.assertSameElements(
        analysis_results.keys(),
        [analyzer.KEY_RUNNING_TIME_DF, analyzer.KEY_NUM_ESTIMABLE_SETS_STATS_DF,
         analyzer.KEY_DESCRIPTION_TO_FILE_DIR])

    # Test if the running time is correct.
    running_time_df = analysis_results[analyzer.KEY_RUNNING_TIME_DF]
    for _, row in running_time_df.iterrows():
      sketch_estimator_name = row[analyzer.SKETCH_ESTIMATOR_COLNAME]
      running_time_file = analysis_results[
          analyzer.KEY_DESCRIPTION_TO_FILE_DIR][
              evaluator.KEY_ESTIMATOR_DIRS][sketch_estimator_name]
      running_time_file = os.path.join(
          running_time_file, evaluator.EVALUATION_RUN_TIME_FILE)
      with open(running_time_file, 'r') as f:
        running_time_from_file = (
            float(f.readline()) / analyzer.RUNNING_TIME_SCALE)
      self.assertEqual(
          row[analyzer.RUNNING_TIME_COLNAME], running_time_from_file,
          f'Running time is not correct: {sketch_estimator_name}.')

    # Test if the number of estimable sets and stats data frame is correct.
    df = analysis_results[analyzer.KEY_NUM_ESTIMABLE_SETS_STATS_DF]
    df_filename = os.path.join(
        analysis_results[
            analyzer.KEY_DESCRIPTION_TO_FILE_DIR][evaluator.KEY_EVALUATION_DIR],
        analyzer.NUM_ESTIMABLE_SETS_FILENAME)
    with open(df_filename, 'r') as f:
      expected = pd.read_csv(f)
    try:
      pd.testing.assert_frame_equal(df, expected)
    except AssertionError:
      self.fail('The number of estimable sets and stats is not correct.')


class FrequencyEstimatorEvaluationAnalyzerTest(absltest.TestCase):

  def test_convert_raw_df_to_long_format(self):
    df = pd.DataFrame({
        analyzer.SKETCH_ESTIMATOR_NAME: ['some_sketch'] * 4,
        analyzer.SCENARIO_NAME: ['some_scenario'] * 4,
        simulator.RUN_INDEX: [0, 0, 1, 1],
        simulator.NUM_SETS: [1, 2, 1, 2],
        simulator.TRUE_CARDINALITY_BASENAME + '1': [10, 20, 10, 20],
        simulator.TRUE_CARDINALITY_BASENAME + '2': [5, 10, 5, 10],
        simulator.ESTIMATED_CARDINALITY_BASENAME + '1': [11, 21, 12, 22],
        simulator.ESTIMATED_CARDINALITY_BASENAME + '2': [4, 9, 3, 8],
    })
    fake_analyzer_class = collections.namedtuple(
        'FrequencyAnalyzer', ['raw_df'])
    fake_analyzer = fake_analyzer_class(raw_df=df)
    df_long = (
        analyzer.FrequencyEstimatorEvaluationAnalyzer
        .convert_raw_df_to_long_format(fake_analyzer))

    expected = pd.DataFrame({
        analyzer.SKETCH_ESTIMATOR_NAME: ['some_sketch'] * 16,
        analyzer.SCENARIO_NAME: ['some_scenario'] * 16,
        simulator.RUN_INDEX: [0, 0, 1, 1] * 4,
        simulator.NUM_SETS: [1, 2] * 8,
        analyzer.CARDINALITY_VALUE: (
            [10, 20, 10, 20, 5, 10, 5, 10]
            + [11, 21, 12, 22, 4, 9, 3, 8]),
        analyzer.CARDINALITY_SOURCE: (
            [simulator.TRUE_CARDINALITY_BASENAME.rstrip('_')] * 8
            + [simulator.ESTIMATED_CARDINALITY_BASENAME.rstrip('_')] * 8),
        analyzer.FREQUENCY_LEVEL: ([1] * 4 + [2] * 4) * 2,
    })

    try:
      pd.testing.assert_frame_equal(df_long, expected)
    except AssertionError:
      self.fail('The long format is not correct.')


if __name__ == '__main__':
  absltest.main()
