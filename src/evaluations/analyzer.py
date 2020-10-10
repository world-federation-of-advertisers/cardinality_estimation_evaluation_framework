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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.common import plotting
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator


ERROR_MARGIN_NAME = 'error_margin'
PROPORTION_OF_RUNS_NAME = 'proportion_of_runs'

ERROR_MARGIN = 0.05
PROPORTION_OF_RUNS = 0.95

RUNNING_TIME_SCALE = 3600  # Convert seconds to hour.

SKETCH_ESTIMATOR_NAME = 'sketch_estimator'
SCENARIO_NAME = 'scenario'
NUM_ESTIMABLE_SETS = 'num_estimable_sets'
RAW_RESULT_DF = 'df'
CARDINALITY_SOURCE = 'source'
CARDINALITY_VALUE = 'cardinality'
FREQUENCY_LEVEL = 'frequency_level'

# The file that summarize the maximum number of sets that can be estimated
# within 5% (or specified by the error_margin) relative error for at least 95%
# (or specified by the proportion_of_runs) runs. It has columns of estimator,
# scenario and num_estimable_sets.
NUM_ESTIMABLE_SETS_FILENAME = 'num_estimable_sets.csv'
BOXPLOT_FILENAME = 'boxplot.png'
BARPLOT_ESTIMABLE_SETS_FILENAME = 'barplot_estimable_sets.png'
BARPLOT_MAX_SETS_FILENAME = 'barplot_max_sets.png'

XLABEL_ROTATE = 'xlabel_rotate'
BOXPLOT_SIZE_WIDTH_INCH = 'boxplot_size_width_inch'
BOXPLOT_SIZE_HEIGHT_INCH = 'boxplot_size_height_inch'
BARPLOT_SIZE_WIDTH_INCH = 'barplot_size_width_inch'
BARPLOT_SIZE_HEIGHT_INCH = 'barplot_size_height_inch'
PLOT_PARAMS = {
    XLABEL_ROTATE: 0,
    BOXPLOT_SIZE_WIDTH_INCH: 12,
    BOXPLOT_SIZE_HEIGHT_INCH: 6,
    BARPLOT_SIZE_WIDTH_INCH: 12,
    BARPLOT_SIZE_HEIGHT_INCH: 6,
}

# Variables related with getting analysis results.
SKETCH_ESTIMATOR_COLNAME = 'sketch_estimator'
RUNNING_TIME_COLNAME = 'running_time'
KEY_DESCRIPTION_TO_FILE_DIR = 'description_to_file_dir'
KEY_NUM_ESTIMABLE_SETS_STATS_DF = 'num_estimable_sets_stats_df'
KEY_RUNNING_TIME_DF = 'running_time_df'


def get_num_estimable_sets(df, num_sets, relative_error, error_margin,
                           proportion_of_runs):
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
      df[df[num_sets] > 1].groupby(num_sets)
      .agg({relative_error: count_estimable}))
  df_estimable = df_estimable.rename(
      columns={relative_error: 'is_estimable'})

  num_of_estimable = 0
  for n in df_estimable.index.values:
    if df_estimable.loc[n, 'is_estimable']:
      num_of_estimable = n
    else:
      break

  return num_of_estimable


class EstimatorEvaluationAnalyzer(object):
  """Analyze the estimator evaluation results."""

  def __init__(self, out_dir, evaluation_directory, evaluation_run_name,
               evaluation_name, estimable_criteria_list,
               plot_params=None):
    """Construct an analyzer.

    Args:
      out_dir: the output directory of analysis results.
      evaluation_directory: the output directory of evaluation results. The
        analyzer will read the evaluation results and output summary tables and
        plots.
      evaluation_run_name: the run name of the evaluation.
      evaluation_name: the name of the evaluation config.
      estimable_criteria_list: a list of tuples of error_margin and
        proportion_of_runs. An error_margin is a positive number setting the
        upper bound of the error, and the proportion_of_runs is a number
        between 0 and 1 that specifies the desired proportion of runs within
        the error margin.
      plot_params: a dictionary of the parameters of plot functions. If not
        given, will use PLOT_PARAMS. Also see PLOT_PARAMS for how it is defined.
    """
    self.estimable_criteria_list = estimable_criteria_list
    if plot_params is None:
      self.plot_params = PLOT_PARAMS
    else:
      self.plot_params = plot_params

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
    num_estimable_sets_stats_df = (
        self.get_relative_error_stats_of_num_of_estimable_sets())
    df_filename = os.path.join(
        self.analysis_file_dirs[evaluator.KEY_EVALUATION_DIR],
        NUM_ESTIMABLE_SETS_FILENAME)
    with open(df_filename, 'w') as f:
      num_estimable_sets_stats_df.to_csv(f, index=False)
    self._save_plot_num_sets_vs_metric()

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
        df[SKETCH_ESTIMATOR_NAME] = estimator_name
        df[SCENARIO_NAME] = scenario_name
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

  def get_num_estimable_sets_df(self):
    """Summarize the number of estimable sets by estimators and scenarios."""
    def _get_num_estimable_sets_series(df, **kwargs):
      return pd.Series({
          NUM_ESTIMABLE_SETS: get_num_estimable_sets(df, **kwargs)})

    df_list = []
    for criteria in self.estimable_criteria_list:
      df = self.raw_df.groupby([SKETCH_ESTIMATOR_NAME, SCENARIO_NAME]).apply(
          _get_num_estimable_sets_series,
          num_sets=simulator.NUM_SETS,
          relative_error=self.error_metric_column,
          error_margin=criteria[0],
          proportion_of_runs=criteria[1]).reset_index()
      df[ERROR_MARGIN_NAME] = criteria[0]
      df[PROPORTION_OF_RUNS_NAME] = criteria[1]
      df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

  def get_relative_error_stats_of_num_of_estimable_sets(self):
    """Get the mean and std of the relative error of the max estimable sets."""
    num_estimable_sets_df = self.get_num_estimable_sets_df()
    df = num_estimable_sets_df.merge(
        right=self.raw_df,
        how='left',
        left_on=[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, NUM_ESTIMABLE_SETS],
        right_on=[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.NUM_SETS])
    result = (
        df.groupby([
            ERROR_MARGIN_NAME, PROPORTION_OF_RUNS_NAME, SKETCH_ESTIMATOR_NAME,
            SCENARIO_NAME, NUM_ESTIMABLE_SETS])
        .agg({self.error_metric_column: ['mean', 'std']}))
    result.columns = result.columns.map('_'.join)
    result = result.reset_index()
    return result

  def _save_plot_num_sets_vs_metric(self):
    """Make and save plots for number of sets versus an arbitrary metric."""
    def plot_one_estimator_under_one_scenario(df):
      ax = plotting.boxplot_relative_error(
          df,
          num_sets=simulator.NUM_SETS,
          relative_error=self.error_metric_column,
          metric_name=self.error_metric_name)
      scenario_name = df[SCENARIO_NAME].values[0]
      estimator_name = df[SKETCH_ESTIMATOR_NAME].values[0]
      ax.set_title(f'{scenario_name}\n{estimator_name}')
      ax.set_ylim(-0.1, 0.1)
      for tick in ax.get_xticklabels():
        tick.set_rotation(self.plot_params[XLABEL_ROTATE])
      # Save the plot to file.
      fig = ax.get_figure()
      plot_file = os.path.join(
          self.analysis_file_dirs[estimator_name][scenario_name],
          BOXPLOT_FILENAME)
      fig.set_size_inches(
          w=self.plot_params[BOXPLOT_SIZE_WIDTH_INCH],
          h=self.plot_params[BOXPLOT_SIZE_HEIGHT_INCH])
      fig.savefig(plot_file)
      plt.close(fig)

    self.raw_df.groupby([SKETCH_ESTIMATOR_NAME, SCENARIO_NAME]).apply(
        plot_one_estimator_under_one_scenario)


class CardinalityEstimatorEvaluationAnalyzer(EstimatorEvaluationAnalyzer):
  """Analyze the cardinality estimator evaluation results."""

  def __init__(self, out_dir, evaluation_directory, evaluation_run_name,
               evaluation_name, estimable_criteria_list,
               plot_params=None):
    self.error_metric_column = simulator.RELATIVE_ERROR_BASENAME + '1'
    self.error_metric_name = 'Relative error'
    super().__init__(out_dir, evaluation_directory, evaluation_run_name,
                     evaluation_name, estimable_criteria_list, plot_params)


class FrequencyEstimatorEvaluationAnalyzer(EstimatorEvaluationAnalyzer):
  """Analyze the frequency estimator evaluation results."""

  def __init__(self, out_dir, evaluation_directory, evaluation_run_name,
               evaluation_name, estimable_criteria_list,
               plot_params=None):
    self.error_metric_column = simulator.SHUFFLE_DISTANCE
    self.error_metric_name = 'Shuffle distance'
    super().__init__(out_dir, evaluation_directory, evaluation_run_name,
                     evaluation_name, estimable_criteria_list, plot_params)

  def __call__(self):
    super().__call__()
    self._save_plot_frequency_distribution_for_report()

  @classmethod
  def _split_source_and_frequency(cls, source_freq):
    """Split the cardinality_frequency string into a named pd.Series.

    For example, true_cardinality_2 will be split into
    pd.Series({CARDINALITY_SOURCE: 'true_cardinality', FREQUENCY_LEVEL: 2}).

    Args:
      source_freq: a string in the format of estimated_cardinality_X or
        true_cardinality_X, where X is an integer.
    Returns:
      A named pd.Series with CARDINALITY_SOURCE and FREQUENCY_LEVEL.
    """
    source_freq_list = source_freq.split('_')
    return pd.Series({
        CARDINALITY_SOURCE: '_'.join(source_freq_list[0:2]),
        FREQUENCY_LEVEL: int(source_freq_list[2]),
    })

  def convert_raw_df_to_long_format(self):
    r"""Convert the raw DataFrame to a long format.

    The raw_df is in a wide format, i.e, it has multiple columns to represent
    the cardinality of different frequency levels. For example, the columns
    look like:
    num_sets,estimated_cardinality_1,estimated_cardinality_2,
    true_cardinality_1,true_cardinality_2\n
    1,11,4,10,5\n
    2,22,8,20,10
    The wide format is difficult to pass to the seaborn plotting functions.
    As such, we convert the raw_df to a long format, whose columns look like:
    num_sets,cardinality_source,cardinality_value,frequency_level\n
    1,'estimated_cardinality',11,1\n
    1,'true_cardinality',10,1\n
    1,'estimated_cardinality',4,2\n
    1,'true_cardinality',5,2\n
    2,'estimated_cardinality',22,1\n
    2,'true_cardinality',20,1\n
    2,'estimated_cardinality',8,2\n
    2,'true_cardinality',10,2

    Returns:
      A pd.DataFrame. It is a long format of self.raw_df, which contains
      columns of SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.RUN_INDEX,
      simulator.NUM_SETS, CARDINALITY_SOURCE, CARDINALITY_VALUE.
    """
    # Get all the columns that are the true or estimated cardinality of all
    # the frequency levels.
    value_vars = []
    for col in self.raw_df.columns:
      if (simulator.TRUE_CARDINALITY_BASENAME in col
          or simulator.ESTIMATED_CARDINALITY_BASENAME in col):
        value_vars.append(col)

    df_long = self.raw_df.melt(
        id_vars=[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.RUN_INDEX,
                 simulator.NUM_SETS],
        value_vars=value_vars,
        var_name=CARDINALITY_SOURCE,
        value_name=CARDINALITY_VALUE,
    )

    df_parsed_source_and_frequency = df_long[CARDINALITY_SOURCE].apply(
        FrequencyEstimatorEvaluationAnalyzer._split_source_and_frequency)

    # Add parsed cardinality source and frequency level columns to the table.
    df_long = pd.concat(
        [df_long[[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.RUN_INDEX,
                  simulator.NUM_SETS, CARDINALITY_VALUE]],
         df_parsed_source_and_frequency],
        axis=1,
    )

    return df_long

  @classmethod
  def _get_per_frequency_cardinality(cls, df, frequency_level_col,
                                     cardinality_col):
    """Calculate per-frequency cardinality.

    For example, given a table of cumulative-frequency cardinality,
       frequency_level  cardinality
                     1           10
                     2            8
                     3            7
    where the cardinality of frequency_level == 1 means that the number of IDs
    that have at least 1 impressions, running this method will return
       frequency_level  cardinality
                     1            2
                     2            1
                     3            7
    where the cardinality of frequency_level == 1 is the number of IDs that have
    frequency equal to 1.

    Args:
      df: a pd.DataFrame that contain columns containing the frequency_level_col
        and the cardinality_col.
      frequency_level_col: a column name in df that represents the the
        frequency level.
      cardinality_col: a column name in df that represents the cardinality.

    Returns:
      A pd.DataFrame containing a column of cardinality_col, and indexed by
      the frequency_level_col. The cardinality is the per-frequency cardinality.
    """
    # Calculate the per frequency level by taking lag-1 difference of the
    # cardinalities of cumulative frequency levels.
    max_frequency = max(df[frequency_level_col])
    per_freq_card = (
        df.set_index(frequency_level_col).sort_index(ascending=False)
        .diff())
    # Fill in the tail: the cardinality of the max_frequency level is the
    # number of IDs with frequency >= max_frequency.
    per_freq_card.loc[max_frequency, cardinality_col] = df.loc[
        df[frequency_level_col] == max_frequency,
        cardinality_col].values[0]
    return per_freq_card.iloc[::-1]

  @classmethod
  def get_per_frequency_cardinality_for_all_cases(cls, df):
    """Calculate per-frequency cardinality for all estimator and scenarios."""
    per_freq_df = (
        df.groupby([
            SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.RUN_INDEX,
            simulator.NUM_SETS, CARDINALITY_SOURCE])
        .apply(
            FrequencyEstimatorEvaluationAnalyzer._get_per_frequency_cardinality,
            frequency_level_col=FREQUENCY_LEVEL,
            cardinality_col=CARDINALITY_VALUE)
        .reset_index()
    )
    return per_freq_df

  def _save_plot_frequency_distribution(self, raw_df_long, filename):
    """Make and save plots for estimated and true frequency distributions.

    Calling this method will create plots in the output directory.

    Args:
      raw_df_long: a long format of pd.DataFrame, which have columns of
        SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, FREQUENCY_LEVEL,
        CARDINALITY_VALUE and CARDINALITY_SOURCE.
      filename: a string of plot filename.
    """
    def plot_one_estimator_under_one_scenario(df):
      ax = plotting.barplot_frequency_distributions(
          df,
          frequency=FREQUENCY_LEVEL,
          cardinality=CARDINALITY_VALUE,
          source=CARDINALITY_SOURCE,
      )
      scenario_name = df[SCENARIO_NAME].values[0]
      estimator_name = df[SKETCH_ESTIMATOR_NAME].values[0]
      num_sets = df[simulator.NUM_SETS].values[0]
      ax.set_title(f'{simulator.NUM_SETS}:{num_sets}\n'
                   f'{scenario_name}\n{estimator_name}')
      # Save the plot to file.
      fig = ax.get_figure()
      plot_file = os.path.join(
          self.analysis_file_dirs[estimator_name][scenario_name],
          filename)
      fig.set_size_inches(
          w=self.plot_params[BARPLOT_SIZE_WIDTH_INCH],
          h=self.plot_params[BARPLOT_SIZE_HEIGHT_INCH])
      fig.savefig(plot_file)
      plt.close(fig)

    # Generate plots for all estimator and scenario combinations.
    raw_df_long.groupby([SKETCH_ESTIMATOR_NAME, SCENARIO_NAME]).apply(
        plot_one_estimator_under_one_scenario)

  def _save_plot_frequency_distribution_num_estimable_sets(self, raw_df_long):
    """Save barplot for comparing estimated vs true frequency distributions.

    This method will save plots for all estimator and scenario combinations.
    The plots include frequency distribution when the number of set is equal to
    the number of estimable sets.

    Args:
      raw_df_long: a long format of pd.DataFrame, which have columns of
        SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, FREQUENCY_LEVEL,
        CARDINALITY_VALUE and CARDINALITY_SOURCE.
    """
    num_estimable_sets_stats_df = (
        self.get_num_estimable_sets_df())
    plot_df = raw_df_long.merge(
        right=num_estimable_sets_stats_df,
        left_on=[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.NUM_SETS],
        right_on=[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, NUM_ESTIMABLE_SETS],
        how='inner',  # If the number of estimable set is 0, then will not plot.
    )
    self._save_plot_frequency_distribution(plot_df,
                                           BARPLOT_ESTIMABLE_SETS_FILENAME)

  def _save_plot_frequency_distribution_max_num_sets(self, raw_df_long):
    """Save barplot for comparing estimated vs true frequency distributions.

    This method will save plots for all estimator and scenario combinations.
    The plots include frequency distribution when the number of set is equal to
    the maximum number of sets.

    Args:
      raw_df_long: a long format of pd.DataFrame, which have columns of
        SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, FREQUENCY_LEVEL,
        CARDINALITY_VALUE and CARDINALITY_SOURCE.
    """
    num_max_sets_df = (
        self.raw_df.groupby([SKETCH_ESTIMATOR_NAME, SCENARIO_NAME])
        .agg({simulator.NUM_SETS: 'max'}).reset_index())
    plot_df = raw_df_long.merge(
        right=num_max_sets_df,
        on=[SKETCH_ESTIMATOR_NAME, SCENARIO_NAME, simulator.NUM_SETS],
        how='inner',
    )
    self._save_plot_frequency_distribution(plot_df,
                                           BARPLOT_MAX_SETS_FILENAME)

  def _save_plot_frequency_distribution_for_report(self):
    """Save barplot for comparing estimated vs true frequency distributions.

    This method will generate plots for all estimator and scenario combinations.
    The plots include frequency distribution when the number of set is equal to
    the number of estimable sets, and when that is equal to the maximum number
    of sets.
    """
    per_freq_df = (
        FrequencyEstimatorEvaluationAnalyzer
        .get_per_frequency_cardinality_for_all_cases(
            self.convert_raw_df_to_long_format()))
    self._save_plot_frequency_distribution_num_estimable_sets(per_freq_df)
    self._save_plot_frequency_distribution_max_num_sets(per_freq_df)


def get_analysis_results(analysis_out_dir, evaluation_run_name,
                         evaluation_name):
  """Get analysis results.

  Args:
    analysis_out_dir: the output folder of the analysis results.
    evaluation_run_name: the run name of the evaluation.
    evaluation_name: the name of the evaluation configuration. For example,
      'smoke_test'.

  Returns:
    A dictionary of the analysis results, which include:
      description_to_file_dir: a dictionary of the analysis results file tree.
      num_estimable_sets_stats_df: a data frame containing the number
        of estimable sets of estimators under different scenarios, and also
        the relative error at the number of estimable sets.
      running_time_df: a data frame containing the running time of each
        sketch_estimator.
  """
  # Read analysis result file tree.
  description_to_file_dir = evaluator.load_directory_tree(
      out_dir=analysis_out_dir,
      run_name=evaluation_run_name,
      evaluation_name=evaluation_name)

  # Read number of estimable sets analysis results.
  filename = os.path.join(
      description_to_file_dir[evaluator.KEY_EVALUATION_DIR],
      NUM_ESTIMABLE_SETS_FILENAME)
  with open(filename, 'r') as f:
    num_estimable_sets_stats_df = pd.read_csv(f)

  # Read running time.
  running_time_df = pd.DataFrame(
      [], columns=[SKETCH_ESTIMATOR_COLNAME, RUNNING_TIME_COLNAME])
  for name, directory in description_to_file_dir[
      evaluator.KEY_ESTIMATOR_DIRS].items():
    filename = os.path.join(directory, evaluator.EVALUATION_RUN_TIME_FILE)
    with open(filename, 'r') as f:
      running_time = float(f.readline())
    running_time_df = running_time_df.append(
        {SKETCH_ESTIMATOR_COLNAME: name,
         RUNNING_TIME_COLNAME: running_time / RUNNING_TIME_SCALE},
        ignore_index=True)
  running_time_df = running_time_df.sort_values(SKETCH_ESTIMATOR_COLNAME)

  return {
      KEY_DESCRIPTION_TO_FILE_DIR: description_to_file_dir,
      KEY_NUM_ESTIMABLE_SETS_STATS_DF: num_estimable_sets_stats_df,
      KEY_RUNNING_TIME_DF: running_time_df
  }
