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
"""Tests for wfa_cardinality_estimation_evaluation_framework.evaluations.evaluator."""

import os

from absl.testing import absltest
import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.estimators import exact_set
from wfa_cardinality_estimation_evaluation_framework.evaluations import configs
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator


class EvaluatorTest(absltest.TestCase):

  def setUp(self):
    super(EvaluatorTest, self).setUp()

    exact_set_lossless = simulator.SketchEstimatorConfig(
        name='exact_set_lossless',
        sketch_factory=exact_set.ExactSet.get_sketch_factory(),
        estimator=exact_set.LosslessEstimator())
    exact_set_less_one = simulator.SketchEstimatorConfig(
        name='exact_set_less_one',
        sketch_factory=exact_set.ExactSet.get_sketch_factory(),
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
                        universe_size=10, num_sets=2, set_size=5))),
            configs.ScenarioConfig(
                name='ind2',
                set_generator_factory=(
                    set_generator.IndependentSetGenerator
                    .get_generator_factory_with_num_and_size(
                        universe_size=10, num_sets=2, set_size=5))),
        ])

    self.run_name = 'test_run'

    def _get_test_evaluator(out_dir, overwrite=False):
      return evaluator.Evaluator(
          evaluation_config=self.evaluation_config,
          sketch_estimator_config_list=self.sketch_estimator_config_list,
          run_name=self.run_name,
          out_dir=out_dir,
          overwrite=overwrite)

    self.get_test_evaluator = _get_test_evaluator

  def test_create_directory_works(self):
    out_dir = self.create_tempdir()
    # Results.
    test_evaluator = self.get_test_evaluator(out_dir)
    # Expected
    eval_dir = os.path.join(
        out_dir, self.run_name,
        self.evaluation_config.name)
    expected = {
        evaluator.KEY_OUT_DIR: out_dir,
        evaluator.KEY_RUN_DIR: os.path.join(out_dir, self.run_name),
        evaluator.KEY_EVALUATION_DIR: eval_dir,
        evaluator.KEY_ESTIMATOR_DIRS: {
            'exact_set_lossless': os.path.join(eval_dir, 'exact_set_lossless'),
            'exact_set_less_one': os.path.join(eval_dir, 'exact_set_less_one')},
        'exact_set_lossless': {
            'ind1': os.path.join(eval_dir, 'exact_set_lossless', 'ind1'),
            'ind2': os.path.join(eval_dir, 'exact_set_lossless', 'ind2')
        },
        'exact_set_less_one': {
            'ind1': os.path.join(eval_dir, 'exact_set_less_one', 'ind1'),
            'ind2': os.path.join(eval_dir, 'exact_set_less_one', 'ind2')
        }
    }

    self.assertEqual(test_evaluator.description_to_file_dir, expected)

  def test_create_directory_prevents_overwrite(self):
    out_dir = self.create_tempdir(name='test_same_dir')
    self.get_test_evaluator(out_dir)
    with self.assertRaises(FileExistsError):
      self.get_test_evaluator(out_dir, overwrite=False)

  def test_create_directory_optionally_allow_overwrite(self):
    out_dir = self.create_tempdir(name='test_same_dir2')
    self.get_test_evaluator(out_dir)
    # Add a random file to check if it is removed.
    test_file_overwrite = os.path.join(out_dir, self.run_name, 'test_file')
    self.create_tempfile(test_file_overwrite)

    try:
      self.get_test_evaluator(out_dir, overwrite=True)
    except FileExistsError:
      self.fail('Doesn\'t successfully overwrite the directory even if opt to.')

    self.assertFalse(os.path.exists(test_file_overwrite),
                     'Does\'t remove the directory before overwrite.')

  def test_load_directory_tree(self):
    # Create directory.
    out_dir = self.create_tempdir('test_load_directory_tree')
    created = evaluator._create_directory_tree(
        run_name=self.run_name,
        evaluation_config=self.evaluation_config,
        sketch_estimator_config_list=self.sketch_estimator_config_list,
        out_dir=out_dir,
        overwrite=False)
    # Load directory.
    loaded = evaluator.load_directory_tree(
        run_name=self.run_name,
        evaluation_name=self.evaluation_config.name,
        out_dir=out_dir)
    self.assertEqual(created, loaded)

  def test_evaluate_all_generate_save_configs(self):
    test_evaluator = self.get_test_evaluator(self.create_tempdir())
    test_evaluator()
    for sketch_estimator_config in self.sketch_estimator_config_list:
      sketch_estimator_config_file = os.path.join(
          test_evaluator.description_to_file_dir[
              evaluator.KEY_ESTIMATOR_DIRS][sketch_estimator_config.name],
          evaluator.ESTIMATOR_CONFIG_FILE)
      self.assertTrue(
          os.path.exists(sketch_estimator_config_file),
          'Estimator config file doesn\'t exist: '
          f'{sketch_estimator_config.name}')

      for scenario_config in self.evaluation_config.scenario_config_list:
        scenario_config_file = os.path.join(
            test_evaluator.description_to_file_dir[
                sketch_estimator_config.name][scenario_config.name],
            evaluator.SCENARIO_CONFIG_FILE)
        self.assertTrue(
            os.path.exists(scenario_config_file),
            f'Scenario config file doesn\'t exist: {scenario_config.name}')

  def test_evaluate_all_save_results(self):
    test_evaluator = self.get_test_evaluator(self.create_tempdir())
    test_evaluator()
    for sketch_estimator_config in self.sketch_estimator_config_list:
      for scenario_config in self.evaluation_config.scenario_config_list:
        scenario_dir = (
            test_evaluator
            .description_to_file_dir[sketch_estimator_config.name][
                scenario_config.name])
        files = [os.path.join(scenario_dir, evaluator.RAW_RESULT_DF_FILENAME),
                 os.path.join(scenario_dir, evaluator.AGG_RESULT_DF_FILENAME)]
        for i in files:
          self.assertTrue(
              os.path.exists(i),
              f'{i} doesn\'t exist in '
              f'{sketch_estimator_config.name}/{scenario_config.name}')

  def test_same_random_state_of_same_scenario(self):
    test_evaluator = self.get_test_evaluator(self.create_tempdir())
    test_evaluator()
    for scenario_config in self.evaluation_config.scenario_config_list:
      # Get the true union cardinality.
      true_cardinalities = []
      for sketch_estimator_config in self.sketch_estimator_config_list:
        df_file = os.path.join(
            test_evaluator.description_to_file_dir[
                sketch_estimator_config.name][scenario_config.name],
            evaluator.RAW_RESULT_DF_FILENAME)
        with open(df_file, 'r') as f:
          df = pd.read_csv(f)
        true_cardinalities.append(
            df[[simulator.RUN_INDEX, simulator.TRUE_CARDINALITY,
                simulator.NUM_SETS]])

      # The true union cardinality should be the same for different estimators
      # under the same scenario.
      df_baseline = true_cardinalities[0]
      for df in true_cardinalities[1:]:
        try:
          pd.testing.assert_frame_equal(df_baseline, df)
        except AssertionError:
          self.fail('Random state not the same for scenario: '
                    f'{scenario_config.name}')


if __name__ == '__main__':
  absltest.main()
