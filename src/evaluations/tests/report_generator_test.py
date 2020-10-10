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
"""Tests for wfa_cardinality_estimation_evaluation_framework.evaluations.tests.report_generator."""
import os
import re

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.estimators import exact_set
from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import configs
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.evaluations import report_generator
from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator


class ReportGeneratorTest(parameterized.TestCase):

  def setUp(self):
    super(ReportGeneratorTest, self).setUp()
    exact_set_lossless = configs.SketchEstimatorConfig(
        name=evaluation_configs.construct_sketch_estimator_config_name(
            sketch_name='exact_set',
            sketch_config='set',
            estimator_name='lossless'),
        sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
        estimator=exact_set.LosslessEstimator())
    exact_set_less_one = configs.SketchEstimatorConfig(
        name=evaluation_configs.construct_sketch_estimator_config_name(
            sketch_name='exact_set',
            sketch_config='set',
            estimator_name='less_one'),
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
                        universe_size=10, num_sets=5, set_size=2))),
            configs.ScenarioConfig(
                name='ind2',
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory_with_num_and_size(
                        universe_size=10, num_sets=5, set_size=2))),
        ])

    self.evaluation_run_name = 'test_run'

    def _run_evaluation_and_simulation(out_dir):
      self.evaluator = evaluator.Evaluator(
          evaluation_config=self.evaluation_config,
          sketch_estimator_config_list=self.sketch_estimator_config_list,
          run_name=self.evaluation_run_name,
          out_dir=out_dir)
      self.evaluator()

      self.analyzer = analyzer.CardinalityEstimatorEvaluationAnalyzer(
          out_dir=out_dir,
          evaluation_directory=out_dir,
          evaluation_run_name=self.evaluation_run_name,
          evaluation_name=self.evaluation_config.name,
          estimable_criteria_list=[(0.05, 0.95), (1.01, 0.9)])
      self.analyzer()

    self.run_evaluation_and_simulation = _run_evaluation_and_simulation

    exact_set_lossless_freq3 = configs.SketchEstimatorConfig(
        name=evaluation_configs.construct_sketch_estimator_config_name(
            sketch_name='exact_set',
            sketch_config='set',
            estimator_name='lossless',
            max_frequency=3),
        sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
        estimator=exact_set.LosslessEstimator(),
        max_frequency=3)
    exact_set_less_one_freq3 = configs.SketchEstimatorConfig(
        name=evaluation_configs.construct_sketch_estimator_config_name(
            sketch_name='exact_set',
            sketch_config='set',
            estimator_name='less_one',
            max_frequency=3),
        sketch_factory=exact_set.ExactMultiSet.get_sketch_factory(),
        estimator=exact_set.LessOneEstimator(),
        sketch_noiser=exact_set.AddRandomElementsNoiser(
            num_random_elements=0, random_state=np.random.RandomState()),
        max_frequency=3)
    self.frequency_sketch_estimator_config_list = (exact_set_lossless_freq3,
                                                   exact_set_less_one_freq3)

    self.frequency_evaluation_run_name = 'freq_test_run'

    def _run_frequency_evaluation_and_simulation(out_dir):
      self.evaluator = evaluator.Evaluator(
          evaluation_config=self.evaluation_config,
          sketch_estimator_config_list=(
              self.frequency_sketch_estimator_config_list),
          run_name=self.frequency_evaluation_run_name,
          out_dir=out_dir)
      self.evaluator()

      self.analyzer = analyzer.FrequencyEstimatorEvaluationAnalyzer(
          out_dir=out_dir,
          evaluation_directory=out_dir,
          evaluation_run_name=self.frequency_evaluation_run_name,
          evaluation_name=self.evaluation_config.name,
          estimable_criteria_list=[(0.05, 0.95), (1.01, 0.9)])
      self.analyzer()

    self.run_frequency_evaluation_and_simulation = (
        _run_frequency_evaluation_and_simulation)

  def test_parse_sketch_estimator_name(self):
    sketch_estimator_name = 'vector_of_counts-4096-sequential-ln3-infty'
    parsed_name = report_generator.ReportGenerator.parse_sketch_estimator_name(
        sketch_estimator_name)
    expected = {
        evaluation_configs.SKETCH: 'vector_of_counts',
        evaluation_configs.SKETCH_CONFIG: '4096',
        evaluation_configs.SKETCH_EPSILON: 'ln3',
        evaluation_configs.ESTIMATOR: 'sequential',
        evaluation_configs.ESTIMATE_EPSILON: 'infty',
    }
    self.assertEqual(parsed_name, expected)

  def test_add_parsed_sketch_estimator_name_cols(self):
    df = pd.DataFrame({
        'sketch_estimator': ['vector_of_counts-4096-sequential-ln3-infty',
                             'bloom_filter-1e6-union_estimator-infty-ln3']})
    result = (
        report_generator.ReportGenerator
        .add_parsed_sketch_estimator_name_cols(df, 'sketch_estimator'))
    expected = pd.DataFrame({
        'sketch_estimator': ['vector_of_counts-4096-sequential-ln3-infty',
                             'bloom_filter-1e6-union_estimator-infty-ln3'],
        evaluation_configs.SKETCH: ['vector_of_counts', 'bloom_filter'],
        evaluation_configs.SKETCH_CONFIG: ['4096', '1e6'],
        evaluation_configs.ESTIMATOR: ['sequential', 'union_estimator'],
        evaluation_configs.SKETCH_EPSILON: ['ln3', 'infty'],
        evaluation_configs.ESTIMATE_EPSILON: ['infty', 'ln3'],
    })
    try:
      pd.testing.assert_frame_equal(result, expected)
    except AssertionError:
      self.fail('Parsed sketch_estimator_name is not added correctly to df.')

  def test_widen_num_estimable_sets_df(self):
    out_dir = self.create_tempdir('test_widen_num_estimable_sets_df')
    self.run_evaluation_and_simulation(out_dir.full_path)
    analysis_results = analyzer.get_analysis_results(
        analysis_out_dir=out_dir.full_path,
        evaluation_run_name=self.evaluation_run_name,
        evaluation_name=self.evaluation_config.name)
    num_estimable_sets_stats_df = (
        report_generator.ReportGenerator.widen_num_estimable_sets_df(
            analysis_results[report_generator.KEY_NUM_ESTIMABLE_SETS_STATS_DF],
            'relative_error_1'))

    # Test values are in correct format.
    regex = re.compile(
        r'\d+<br>relative_error: mean=(((-)?\d+\.\d+)|(nan)), '
        r'std=(((-)?\d+\.\d+)|(nan))')
    for s in np.ndarray.flatten(num_estimable_sets_stats_df.values):
      self.assertRegex(s, regex, f'value {s} not is not in correct format.')

    # Test the columns are correct.
    regex = r'(\d+)\%\/(\d+)'
    for col in num_estimable_sets_stats_df.columns.values:
      self.assertRegex(
          col[0], regex, f'column {col[0]} not is not in correct format.')

  @parameterized.parameters(
      (report_generator.ANALYSIS_TYPE_CARDINALITY,),
      (report_generator.ANALYSIS_TYPE_FREQUENCY,),
  )
  def test_generate_plots_html(self, analysis_type):
    out_dir = self.create_tempdir('test_generate_plots_html_' + analysis_type)
    if analysis_type == report_generator.ANALYSIS_TYPE_CARDINALITY:
      self.run_evaluation_and_simulation(out_dir.full_path)
      sketch_estimator_list = [
          i.name for i in self.sketch_estimator_config_list]
      run_name = self.evaluation_run_name
    else:
      self.run_frequency_evaluation_and_simulation(out_dir.full_path)
      sketch_estimator_list = [
          i.name for i in self.frequency_sketch_estimator_config_list]
      run_name = self.frequency_evaluation_run_name

    analysis_results = analyzer.get_analysis_results(
        analysis_out_dir=out_dir.full_path,
        evaluation_run_name=run_name,
        evaluation_name=self.evaluation_config.name)
    # Generate plots html.
    description_to_file_dir = analysis_results[
        report_generator.KEY_DESCRIPTION_TO_FILE_DIR]
    scenario_list = [
        conf.name for conf in self.evaluation_config.scenario_config_list]
    plot_html = report_generator.ReportGenerator.generate_plots_html(
        description_to_file_dir=description_to_file_dir,
        sketch_estimator_list=sketch_estimator_list,
        scenario_list=scenario_list,
        out_dir=out_dir.full_path,
        analysis_type=analysis_type)
    # Read the table from html.
    plot_html = ' '.join(plot_html.split('\n'))
    regex = r'<table(.+?)</table>'
    for h in re.finditer(regex, plot_html):
      tab = pd.read_html(h.group(0), header=[0, 1])[0]
      self.assertGreater(tab.shape[0], 0,
                         'The html table is empty table.')
      self.assertIn(f'<img src="{run_name}/test_evaluation', h.group(0),
                    'Table link is missing.')

  def test_generate_and_save_html_cardinality_report(self):
    analysis_out_dir = self.create_tempdir('analysis_dir')
    report_out_dir = self.create_tempdir('test_report_dir')
    self.run_evaluation_and_simulation(analysis_out_dir.full_path)
    new_report = report_generator.CardinalityReportGenerator(
        out_dir=report_out_dir.full_path,
        analysis_out_dir=analysis_out_dir.full_path,
        evaluation_run_name=self.evaluation_run_name,
        evaluation_name=self.evaluation_config.name)
    report_url = new_report('new_report')
    self.assertTrue(os.path.exists(report_url))

  def test_generate_and_save_html_frequency_report(self):
    analysis_out_dir = self.create_tempdir('analysis_dir')
    report_out_dir = self.create_tempdir('test_report_dir')
    self.run_frequency_evaluation_and_simulation(analysis_out_dir.full_path)
    new_report = report_generator.FrequencyReportGenerator(
        out_dir=report_out_dir.full_path,
        analysis_out_dir=analysis_out_dir.full_path,
        evaluation_run_name=self.frequency_evaluation_run_name,
        evaluation_name=self.evaluation_config.name)
    report_url = new_report('new_report')
    self.assertTrue(os.path.exists(report_url))


if __name__ == '__main__':
  absltest.main()
