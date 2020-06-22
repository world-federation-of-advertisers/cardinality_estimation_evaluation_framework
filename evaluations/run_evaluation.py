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
r"""A commandline tool to run evaluations of cardinality estimators.

To run this evaluation, in command line, type:
python run_evaluation.py \
--evaluation_out_dir="evaluation_output" \
--analysis_out_dir="analysis_output" \
--evaluation_config="smoke_test" \
--sketch_estimator_configs="vector_of_counts-4096-ln3-sequential" \
--evaluation_run_name="simple_run"
"""

from absl import app
from absl import flags
from absl import logging

from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from data import evaluation_configs


FLAGS = flags.FLAGS

# Directories.
flags.DEFINE_string('evaluation_out_dir', None,
                    'The output directory of evaluation results.')
flags.DEFINE_string('analysis_out_dir', None,
                    'The output directory of analysis results.')

# Configs.
flags.DEFINE_enum(
    'evaluation_config', None, evaluation_configs.EVALUATION_CONFIG_NAMES,
    'The name of the evaluation configuration. '
    'See evaluation_configs.EVALUATION_CONFIG_NAMES for the complete list '
    'of supported configs.')
flags.DEFINE_multi_enum(
    'sketch_estimator_configs', None,
    evaluation_configs.ESTIMATOR_CONFIG_NAMES,
    'The name of the estimator configuration documented in '
    'evaluation_configs.ESTIMATOR_CONFIG_NAMES. '
    'Can evaluate multiple estimator_config.')
flags.DEFINE_string(
    'evaluation_run_name', None,
    'The name of this evaluation run.')
flags.DEFINE_integer(
    'num_runs', None, 'The number of runs per scenario.', lower_bound=1)

# Analysis parameters.
flags.DEFINE_float(
    'error_margin', 0.05,
    'a positive number setting the upper bound of the error. '
    'By default, set to 0.05.',
    lower_bound=0)
flags.DEFINE_float(
    'proportion_of_runs', 0.95,
    'a number between 0 and 1 that specifies the proportion of runs. '
    'By default, set to 0.95.',
    lower_bound=0, upper_bound=1)
flags.DEFINE_integer('boxplot_size_width_inch', 12,
                     'The widths of the boxplot in inches.')
flags.DEFINE_integer('boxplot_size_height_inch', 6,
                     'The widths of the boxplot in inches.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  required_flags = (
      'evaluation_out_dir', 'analysis_out_dir',
      'evaluation_config', 'sketch_estimator_configs', 'evaluation_run_name',
      'num_runs')
  for f in required_flags:
    flags.mark_flag_as_required(f)

  logging.set_verbosity(logging.INFO)
  logging.info('====Running %s using evaluation %s for:\n%s',
               FLAGS.evaluation_config,
               FLAGS.evaluation_run_name,
               ', '.join(FLAGS.sketch_estimator_configs))

  evaluation_config = evaluation_configs.NAME_TO_EVALUATION_CONFIGS[
      FLAGS.evaluation_config](FLAGS.num_runs)
  sketch_estimator_config_list = [
      evaluation_configs.NAME_TO_ESTIMATOR_CONFIGS[conf]
      for conf in FLAGS.sketch_estimator_configs]

  # Run simulations.
  generate_results = evaluator.Evaluator(
      evaluation_config=evaluation_config,
      sketch_estimator_config_list=sketch_estimator_config_list,
      run_name=FLAGS.evaluation_run_name,
      out_dir=FLAGS.evaluation_out_dir)
  generate_results()

  # Analyze results.
  logging.info('====Analyzing the results.')
  generate_summary = analyzer.CardinalityEstimatorEvaluationAnalyzer(
      out_dir=FLAGS.analysis_out_dir,
      evaluation_directory=FLAGS.evaluation_out_dir,
      evaluation_run_name=FLAGS.evaluation_run_name,
      evaluation_name=evaluation_config.name,
      error_margin=FLAGS.error_margin,
      proportion_of_runs=FLAGS.proportion_of_runs,
      plot_params={
          analyzer.BOXPLOT_SIZE_WIDTH_INCH: FLAGS.boxplot_size_width_inch,
          analyzer.BOXPLOT_SIZE_HEIGHT_INCH: FLAGS.boxplot_size_height_inch,
      })
  generate_summary()

  logging.info('====Evaluation and analysis done!')

if __name__ == '__main__':
  app.run(main)
