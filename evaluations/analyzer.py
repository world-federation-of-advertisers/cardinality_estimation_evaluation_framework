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

"""Analyze the simulation results."""
import os
import shutil

import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.common import plotting
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator


ERROR_MARGIN = 0.05
PROPORTION_OF_RUNS = 0.95

ESTIMATOR_NAME = 'estimator'
SCENARIO_NAME = 'scenario'
NUM_ESTIMABLE_SETS = 'num_estimable_sets'
RAW_RESULT_DF = 'df'

# The file that summarize the maximum number of sets that can be estimated
# within 5% (or specified by the error_margin) relative error for at least 95%
# (or specified by the proportion_of_runs) runs. It has columns of estimator,
# scenario and num_estimable_sets.
NUM_ESTIMABLE_SETS_FILENAME = 'num_estimable_sets.csv'
BOXPLOT_FILENAME = 'boxplot.png'

BOXPLOT_SIZE_WIDTH_INCH = 'boxplot_size_width_inch'
BOXPLOT_SIZE_HEIGHT_INCH = 'boxplot_size_width_inch'
PLOT_PARAMS = {
    BOXPLOT_SIZE_WIDTH_INCH: 12,
    BOXPLOT_SIZE_HEIGHT_INCH: 6,
}


def get_num_estimable_sets(df, num_sets=simulator.NUM_SETS,
                           relative_error=simulator.RELATIVE_ERROR,
                           error_margin=ERROR_MARGIN,
                           proportion_of_runs=PROPORTION_OF_RUNS):
  """Get the number of estimable sets.

  For example, set error_margin = 0.05 and proportion_of_runs = 0.95. Then
  the number of estimable sets is defined as the number of sets whose union
  cardinality can be estimated such that 95% of the runs are within a 5%
  relative error.

  Args:
    df: a pd.DataFrame that have columns of num_sets and relative_error.
    num_sets: a column name in df that specifies the number of sets.
    relative_error: a column name in df that specifies the relative error.
    error_margin: a positive number setting the upper bound of the error. By
      default, set to 0.05.
    proportion_of_runs: a number between 0 and 1 that specifies the proportion
      of runs. By default, set to 0.95.

  Returns:
    The number of estimable sets.
  """
  if not set([num_sets, relative_error]).issubset(df.columns):
    raise ValueError(f'{num_sets} or {relative_error} not found in df.')

  def count_estimable(e):
    return np.mean(np.abs(e) < error_margin) >= proportion_of_runs

  df_estimable = (
      df.groupby(num_sets).agg({relative_error: count_estimable}))
  df_estimable = df_estimable.rename(
      columns={relative_error: 'is_estimable'})

  num_of_estimable = 0
  for n in df_estimable.index.values:
    if df_estimable.loc[n, 'is_estimable']:
      num_of_estimable = n
    else:
      break

  return num_of_estimable


class CardinalityEstimatorEvaluationAnalyzer(object):
  """Analyze the cardinality estimator evaluation results."""

  def __init__(self, out_dir, evaluation_directory, evaluation_run_name,
               evaluation_name, error_margin=ERROR_MARGIN,
               proportion_of_runs=PROPORTION_OF_RUNS,
               plot_params=None):
    """Construct an analyzer.

    Args:
      out_dir: the output directory of analysis results.
      evaluation_directory: the output directory of evaluation results. The
        analyzer will read the evaluation results and output summary tables and
        plots.
      evaluation_run_name: the run name of the evaluation.
      evaluation_name: the name of the evaluation config.
      error_margin: a positive number setting the upper bound of the error. By
        default, set to 0.05.
      proportion_of_runs: a number between 0 and 1 that specifies the desired
        proportion of runs within the error margin. By default, set to 0.95.
      plot_params: a dictionary of the parameters of plot functions. If not
        given, will use PLOT_PARAMS. Also see PLOT_PARAMS for how it is defined.
    """
    self.error_margin = error_margin
    self.proportion_of_runs = proportion_of_runs
    self.plot_params = plot_params or PLOT_PARAMS

    # Get all the raw results.
    self.evaluation_file_dirs = evaluator.load_directory_tree(
        out_dir=evaluation_directory,
        run_name=evaluation_run_name,
        evaluation_name=evaluation_name)
    self.raw_df = (
        CardinalityEstimatorEvaluationAnalyzer
        .read_evaluation_results(self.evaluation_file_dirs))

    # Create the analysis directory.
    if out_dir is None:
      out_dir = os.getcwd()
    if out_dir != evaluation_directory:
      shutil.copytree(
          self.evaluation_file_dirs[evaluator.KEY_RUN_DIR],
          os.path.join(out_dir, evaluation_run_name))
    self.analysis_file_dirs = evaluator.load_directory_tree(
        out_dir=out_dir,
        run_name=evaluation_run_name,
        evaluation_name=evaluation_name)

  def __call__(self):
    num_estimable_sets_df = self.get_num_estimable_sets_df()
    df_filename = os.path.join(
        self.analysis_file_dirs[evaluator.KEY_EVALUATION_DIR],
        NUM_ESTIMABLE_SETS_FILENAME)
    with open(df_filename, 'w') as f:
      num_estimable_sets_df.to_csv(f, index=False)
    self.save_plot_num_sets_vs_relative_error()

  @classmethod
  def read_evaluation_results(cls, file_dirs):
    """Read evaluation results.

    Args:
      file_dirs: a dictionary of file directories of the evaluation which is
        generated by the create_directory method of evaluator.Evaluation.

    Returns:
      A pandas.DataFrame containing columns of the estimator name, the scenario
      name, and the corresponding raw evaluation result data frame.
    """
    df_list = []
    for estimator_name in file_dirs[evaluator.KEY_ESTIMATOR_DIRS].keys():
      for scenario_name in file_dirs[estimator_name].keys():
        df_file = os.path.join(
            file_dirs[estimator_name][scenario_name],
            evaluator.RAW_RESULT_DF_FILENAME)
        with open(df_file, 'r') as f:
          df = pd.read_csv(f)
        df_list.append((estimator_name, scenario_name, df))

    return pd.DataFrame(
        df_list, columns=[ESTIMATOR_NAME, SCENARIO_NAME, RAW_RESULT_DF])

  def get_num_estimable_sets_df(self):
    """Summarize the number of estimable sets by estimators and scenarios."""

    def f(x):
      num_estimable_sets = get_num_estimable_sets(
          x[RAW_RESULT_DF], num_sets=simulator.NUM_SETS,
          relative_error=simulator.RELATIVE_ERROR,
          error_margin=self.error_margin,
          proportion_of_runs=self.proportion_of_runs)
      return pd.Series({
          ESTIMATOR_NAME: x[ESTIMATOR_NAME],
          SCENARIO_NAME: x[SCENARIO_NAME],
          NUM_ESTIMABLE_SETS: num_estimable_sets})

    return self.raw_df.apply(f, axis=1)

  def save_plot_num_sets_vs_relative_error(self):
    """Make and save plots for number of sets versus relative error."""
    def f(x):
      # Make a plot.
      ax = plotting.boxplot_relative_error(
          x[RAW_RESULT_DF],
          num_sets=simulator.NUM_SETS,
          relative_error=simulator.RELATIVE_ERROR)
      ax.set_title(f'{x[SCENARIO_NAME]}\n{x[ESTIMATOR_NAME]}')
      ax.set_ylim(-0.1, 0.1)
      for tick in ax.get_xticklabels():
        tick.set_rotation(90)
      # Save the plot to file.
      fig = ax.get_figure()
      plot_file = os.path.join(
          self.analysis_file_dirs[x[ESTIMATOR_NAME]][x[SCENARIO_NAME]],
          BOXPLOT_FILENAME)
      fig.set_size_inches(
          w=self.plot_params[BOXPLOT_SIZE_WIDTH_INCH],
          h=self.plot_params[BOXPLOT_SIZE_HEIGHT_INCH])
      fig.savefig(plot_file)

    self.raw_df.apply(f, axis=1)
