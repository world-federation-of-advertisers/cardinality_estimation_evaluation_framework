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
  wfa-run-evaluation \
    --evaluation_out_dir="evaluation_output" \
    --analysis_out_dir="analysis_output" \
    --evaluation_config="smoke_test" \
    --sketch_estimator_configs="vector_of_counts-4096-ln3-sequential" \
    --evaluation_run_name="simple_run" \
    --num_runs=1
"""

from absl import app
from absl import flags
from absl import logging

from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.evaluations import report_generator
from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs


FLAGS = flags.FLAGS

flags.DEFINE_bool('run_evaluation', True, 'Run evaluation or not.')
flags.DEFINE_bool('run_analysis', True, 'Run analysis or not.')
flags.DEFINE_bool('generate_html_report', True,
                  'Generate an HTML report or not.')

# Directories.
flags.DEFINE_string('evaluation_out_dir', None,
                    'The output directory of evaluation results.')
flags.DEFINE_string('analysis_out_dir', None,
                    'The output directory of analysis results.')
flags.DEFINE_string('report_out_dir', None,
                    'The output directory of analysis results.')

# Configs.
flags.DEFINE_string('evaluation_config', None,
                    'The name of the evaluation configuration. ')
flags.DEFINE_multi_string(
    'sketch_estimator_configs', None,
    'The name(s) of the estimator configuration(s)')
flags.DEFINE_string('evaluation_run_name', None,
                    'The name of this evaluation run.')
flags.DEFINE_integer('num_runs', None,
                     'The number of runs per scenario.', lower_bound=1)
flags.DEFINE_integer(
    'num_workers', 0,
    'The number of processes to use in parallel. If 1, runs serially.'
    'If 0 or less, use as many processes as cores.')

# Analysis parameters.
flags.DEFINE_list(
    'error_margin', '0.05',
    'A comma-separated list of positive numbers setting the upper bound of '
    'the error. By default, use 0.05.')
flags.DEFINE_list(
    'proportion_of_runs', '0.95',
    'A comma-separated list of a number between 0 and 1 that specifies the '
    'proportion of runs. By default, use 0.95.')
flags.DEFINE_integer('boxplot_xlabel_rotate', 0,
                     'The degrees of rotation of the x labels in the boxplot.')
flags.DEFINE_integer('boxplot_size_width_inch', 12,
                     'The widths of the boxplot in inches.')
flags.DEFINE_integer('boxplot_size_height_inch', 6,
                     'The height of the boxplot in inches.')
flags.DEFINE_integer('barplot_size_width_inch', 12,
                     'The widths of the barplot in inches.')
flags.DEFINE_integer('barplot_size_height_inch', 6,
                     'The height of the barplot in inches.')
flags.DEFINE_enum('analysis_type', 'cardinality', ['cardinality', 'frequency'],
                  'Type of analysis that is to be performed.')
flags.DEFINE_integer('max_frequency', 10, 'Maximum frequency to be analyzed.')


required_flags = ('evaluation_config', 'sketch_estimator_configs',
                  'evaluation_run_name', 'num_runs', 'evaluation_out_dir',
                  'analysis_out_dir')


def _run(run_evaluation, run_analysis, generate_html_report, evaluation_out_dir,
         analysis_out_dir, report_out_dir, evaluation_config,
         sketch_estimator_configs, evaluation_run_name, num_runs,
         num_workers, error_margin, proportion_of_runs,
         boxplot_xlabel_rotate, boxplot_size_width_inch,
         boxplot_size_height_inch, analysis_type,
         max_frequency, barplot_size_width_inch=None,
         barplot_size_height_inch=None):
  """Run evaluation."""
  evaluation_config = evaluation_configs.get_evaluation_config(
      evaluation_config)(num_runs)

  sketch_estimator_config_list = evaluation_configs.get_estimator_configs(
      sketch_estimator_configs, max_frequency)

  if run_evaluation:
    logging.info('====Running %s using evaluation %s for:\n%s',
                 evaluation_config,
                 evaluation_run_name,
                 ', '.join(sketch_estimator_configs))
    generate_results = evaluator.Evaluator(
        evaluation_config=evaluation_config,
        sketch_estimator_config_list=sketch_estimator_config_list,
        run_name=evaluation_run_name,
        out_dir=evaluation_out_dir,
        workers=num_workers)
    generate_results()

  error_margin = [float(x) for x in error_margin]
  proportion_of_runs = [float(x) for x in proportion_of_runs]
  estimable_criteria_list = list(map(tuple, zip(error_margin,
                                                proportion_of_runs)))

  if analysis_type == 'frequency':
    estimator_analyzer_func = analyzer.FrequencyEstimatorEvaluationAnalyzer
    report_generator_func = report_generator.FrequencyReportGenerator
  else:
    estimator_analyzer_func = analyzer.CardinalityEstimatorEvaluationAnalyzer
    report_generator_func = report_generator.CardinalityReportGenerator

  if run_analysis:
    logging.info('====Analyzing the results.')
    generate_summary = estimator_analyzer_func(
        out_dir=analysis_out_dir,
        evaluation_directory=evaluation_out_dir,
        evaluation_run_name=evaluation_run_name,
        evaluation_name=evaluation_config.name,
        estimable_criteria_list=estimable_criteria_list,
        plot_params={
            analyzer.XLABEL_ROTATE: boxplot_xlabel_rotate,
            analyzer.BOXPLOT_SIZE_WIDTH_INCH: boxplot_size_width_inch,
            analyzer.BOXPLOT_SIZE_HEIGHT_INCH: boxplot_size_height_inch,
            analyzer.BARPLOT_SIZE_WIDTH_INCH: barplot_size_width_inch,
            analyzer.BARPLOT_SIZE_HEIGHT_INCH: barplot_size_height_inch,
        })
    generate_summary()

  logging.info('====Evaluation and analysis done!')

  if generate_html_report:
    generate_report = report_generator_func(
        out_dir=report_out_dir,
        analysis_out_dir=analysis_out_dir,
        evaluation_run_name=evaluation_run_name,
        evaluation_name=evaluation_config.name)
    report_url = generate_report()
    logging.info('====Report generated: %s.', report_url)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.set_verbosity(logging.INFO)

  _run(
      run_evaluation=FLAGS.run_evaluation,
      run_analysis=FLAGS.run_analysis,
      generate_html_report=FLAGS.generate_html_report,
      evaluation_out_dir=FLAGS.evaluation_out_dir,
      analysis_out_dir=FLAGS.analysis_out_dir,
      report_out_dir=FLAGS.report_out_dir,
      evaluation_config=FLAGS.evaluation_config,
      sketch_estimator_configs=FLAGS.sketch_estimator_configs,
      evaluation_run_name=FLAGS.evaluation_run_name,
      num_runs=FLAGS.num_runs,
      num_workers=FLAGS.num_workers,
      error_margin=FLAGS.error_margin,
      proportion_of_runs=FLAGS.proportion_of_runs,
      boxplot_xlabel_rotate=FLAGS.boxplot_xlabel_rotate,
      boxplot_size_width_inch=FLAGS.boxplot_size_width_inch,
      boxplot_size_height_inch=FLAGS.boxplot_size_height_inch,
      barplot_size_width_inch=FLAGS.barplot_size_width_inch,
      barplot_size_height_inch=FLAGS.barplot_size_height_inch,
      analysis_type=FLAGS.analysis_type,
      max_frequency=FLAGS.max_frequency,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(required_flags)
  app.run(main)
