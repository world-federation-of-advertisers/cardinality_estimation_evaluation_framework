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
"""Generate evaluation report."""

import os
import shutil

import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.evaluations.analyzer import KEY_DESCRIPTION_TO_FILE_DIR
from wfa_cardinality_estimation_evaluation_framework.evaluations.analyzer import KEY_NUM_ESTIMABLE_SETS_STATS_DF
from wfa_cardinality_estimation_evaluation_framework.evaluations.analyzer import KEY_RUNNING_TIME_DF
from wfa_cardinality_estimation_evaluation_framework.evaluations.data import evaluation_configs
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator

pd.set_option('display.max_colwidth', None)

HTML_REPORT_FILENAME = 'cardinality_estimator_evaluation_report.html'

ESTIMABLE_CRITERIA_COLNAME = 'estimable_criteria'

MESSAGE_HTML_TEMPLATE = """
    <html>
      <head>
        <title>
        Reach Cardinality Estimators Evaluation
        </title>
      </head>
      <body>
        <h1>Reach Cardinality Estimators Evaluation Report</h1>
          <h2>Summary</h2>
            <h3>Maximum number of estimable sets and the corresponding relative
            errors</h3>
              <p>
                The larger the number of estimable sets is, the better the
                cardinality estimator is.
              </p>
              {num_estimable_sets_stats_df_html}
            <h3>Running time</h3>
              <p>
                The nunning time of a cardinality estimator over all the
                scenarios in hours. The smaller the better.<br>
                {running_time_df_html}
              </p>
          <h2>Scenarios<h2>
            {plot_df_html}
      </body>
    </html>"""

# Number of digits that should be shown when reporting the relative error
RELATIVE_ERROR_FORMAT_ACCURACY = 4

class ReportGenerator:
  """Generate HTML report for an estimator evaluation."""
  
  def __init__(self, out_dir, analysis_out_dir, evaluation_run_name,
               evaluation_name):
    """Read analysis results and generate HTML report.

    Args:
      out_dir: the output direcotry of the report.
      analysis_out_dir: the output folder of the analysis results.
      evaluation_run_name: the run name of the evaluation.
      evaluation_name: the name of the evaluation configuration. For example,
        'smoke_test'.
    """
    if out_dir is None:
      out_dir = os.getcwd()
    self.out_dir = out_dir

    # Copy the analysis results to the report output directory, so that the HTML
    # report can be correctly rendered even if we move the csv files, plots,
    # etc.
    if out_dir != analysis_out_dir:
      analysis_file_dirs = evaluator.load_directory_tree(
          out_dir=analysis_out_dir,
          run_name=evaluation_run_name,
          evaluation_name=evaluation_name)
      shutil.copytree(
          analysis_file_dirs[evaluator.KEY_RUN_DIR],
          os.path.join(out_dir, evaluation_run_name))

    self.analysis_results = analyzer.get_analysis_results(
        out_dir, evaluation_run_name, evaluation_name)

    self.analysis_results[KEY_NUM_ESTIMABLE_SETS_STATS_DF] = (
        ReportGenerator.add_parsed_sketch_estimator_name_cols(
            self.analysis_results[KEY_NUM_ESTIMABLE_SETS_STATS_DF],
            analyzer.SKETCH_ESTIMATOR_NAME))

  def __call__(self, out_filename=HTML_REPORT_FILENAME):
    html_report = self.generate_html_report()
    html_report_filename = os.path.join(self.out_dir, out_filename)
    with open(html_report_filename, 'w') as f:
      f.write(html_report)
    return html_report_filename

  @classmethod
  def parse_sketch_estimator_name(cls, sketch_estimator_name):
    """Parse a sketch estimator name.

    Args:
      sketch_estimator_name: a string of sketch estimator name that conforms to
        the format of sketch_name-sketch_config-epsilon-estimator_name.

    Returns:
      A dictionary of sketch_name, sketch_config, epsilon, and estimator_name.
    """
    keys = evaluation_configs.SKETCH_ESTIMATOR_CONFIG_NAMES_FORMAT
    values = sketch_estimator_name.split('-')
    return {key: value for key, value in zip(keys, values)}

  @classmethod
  def add_parsed_sketch_estimator_name_cols(cls, df, sketch_estimator_name_col):
    """Add parsed sketch and estimator name to the data frame.

    Args:
      df: a DataFrame that contains a column of sketch_estimator_name.
      sketch_estimator_name_col: the colname of the sketch_estimator_name.

    Returns:
      A DataFrame with additional columns of the sketch name, sketch config,
      epsilon and estimator name.
    """
    names_df = df[sketch_estimator_name_col].apply(
        lambda x: pd.Series(ReportGenerator.parse_sketch_estimator_name(x)))
    return pd.concat([df, names_df], axis=1)

  @classmethod
  def widen_num_estimable_sets_df(cls, df, error_metric_column):
    """Widen the number of estimable sets with statistics df.

    Args:
      df: a data frame that contains the number of estimable sets by estimators
        and scenarios, and the error metric stats at the number of estimable
        sets.
      error_metric_column: name of column containing the error metric to be
        used for widening.

    Returns:
      A wide format of the number of estimable sets and the relative error
      stats.
    """
    def _format_num_estimable_sets_df(df, error_metric_column):
      df = df.copy()
      df['num_estimable_sets_cell'] = (
          df[analyzer.NUM_ESTIMABLE_SETS].astype('str')
          + '<br>relative_error: mean='
          + df[error_metric_column + '_mean'].round(
            RELATIVE_ERROR_FORMAT_ACCURACY).astype('str')
          + ', std='
          + df[error_metric_column + '_std'].round(
            RELATIVE_ERROR_FORMAT_ACCURACY).astype('str')
      )
      df[ESTIMABLE_CRITERIA_COLNAME] = (
          df[analyzer.PROPORTION_OF_RUNS_NAME].apply('{0:.0%}'.format)
          + '/'
          + df[analyzer.ERROR_MARGIN_NAME].apply('{0:.0%}'.format))
      return df

    df = _format_num_estimable_sets_df(df, error_metric_column)
    return df.pivot_table(
        values='num_estimable_sets_cell',
        index=analyzer.SCENARIO_NAME,
        columns=[ESTIMABLE_CRITERIA_COLNAME, analyzer.SKETCH_ESTIMATOR_NAME],
        aggfunc=''.join)

  @classmethod
  def generate_boxplot_html(cls, sketch_estimator_list, scenario_list,
                            description_to_file_dir, out_dir):
    """Generate HTML for boxplot of relative errors by scenarios.

    Args:
      sketch_estimator_list: a list of sketch_estimator names, which conforms to
        sketch-sketch_config-epsilon-estimator.
      scenario_list: a list of scenario names.
      description_to_file_dir: a dictionary that contains the file tree of the
        analysis results.
      out_dir: a string representing the output directory.

    Returns:
      The HTML of the plot if the plot file exists. Otherwise, return None.
    """
    sketch_estimator_df = pd.DataFrame(
        [], columns=evaluation_configs.SKETCH_ESTIMATOR_CONFIG_NAMES_FORMAT)
    for sketch_estimator in sketch_estimator_list:
      sketch_estimator_df = sketch_estimator_df.append(
          ReportGenerator.parse_sketch_estimator_name(sketch_estimator),
          ignore_index=True)

    def _get_boxplot_html(row):
      sketch_estimator_name = '-'.join([
          row[i] for i in (
              evaluation_configs.SKETCH_ESTIMATOR_CONFIG_NAMES_FORMAT)
      ])
      if sketch_estimator_name not in description_to_file_dir[
          evaluator.KEY_ESTIMATOR_DIRS]:
        return None

      estimator_dir = description_to_file_dir[sketch_estimator_name]
      if row[analyzer.SCENARIO_NAME] not in estimator_dir:
        return None

      img_file = os.path.join(
          estimator_dir[row[analyzer.SCENARIO_NAME]],
          analyzer.BOXPLOT_FILENAME)
      # Get relative path.
      img_file = os.path.relpath(img_file, out_dir)

      title = (
          row[evaluation_configs.SKETCH] + ', '
          + row[evaluation_configs.ESTIMATOR]  + '<br>'
          + 'sketch_config: ' + row[evaluation_configs.SKETCH_CONFIG] + '<br>'
          + 'epsilon: ' + row[evaluation_configs.EPSILON])

      return (
          '<figure>'
          f'<figcaption>{title}</figcaption>'
          f'<img src="{img_file}" width="400" height="250">'
          '</figure>'
      )

    # Get plots from all the scenarios.
    plot_df_html_list = []
    for scenario in scenario_list:
      plot_df = sketch_estimator_df.copy()
      plot_df[analyzer.SCENARIO_NAME] = scenario
      plot_df['link'] = plot_df.apply(_get_boxplot_html, axis=1)
      plot_df = plot_df.pivot_table(
          values='link',
          index=evaluation_configs.EPSILON,
          columns=evaluation_configs.SKETCH,
          aggfunc=''.join)
      plot_df_html_list.append(
          f'<h3>{scenario}</h3>\n' + plot_df.to_html(escape=False))

    return '\n'.join(plot_df_html_list)

  def generate_html_report(self):
    """Generate HTML report."""
    # Generate the number of estimable sets html tables by epsilon.
    epsilon_list = (
        self.analysis_results[KEY_NUM_ESTIMABLE_SETS_STATS_DF]['epsilon']
        .unique())
    num_estimable_sets_stats_df_html_list = []
    df = self.analysis_results[KEY_NUM_ESTIMABLE_SETS_STATS_DF]
    for epsilon in epsilon_list:
      one_html = ReportGenerator.widen_num_estimable_sets_df(
          df.loc[df['epsilon'] == epsilon], self.error_metric_column
      ).to_html(escape=False)
      one_html = f'<h4>epsilon={epsilon}</h4><p>' + one_html + '</p>'
      num_estimable_sets_stats_df_html_list.append(one_html)
    num_estimable_sets_stats_df_html = '\n'.join(
        num_estimable_sets_stats_df_html_list)

    # Generate the running time html table.
    running_time_df_html = (
        self.analysis_results[KEY_RUNNING_TIME_DF].to_html(escape=False))

    # Generate the boxplot by scenarios.
    sketch_estimator_list = self.analysis_results[
        KEY_DESCRIPTION_TO_FILE_DIR][evaluator.KEY_ESTIMATOR_DIRS].keys()
    scenario_list = set()
    for estimator in sketch_estimator_list:
      scenario_list.update(self.analysis_results[
          KEY_DESCRIPTION_TO_FILE_DIR][estimator].keys())
    plot_df_html = ReportGenerator.generate_boxplot_html(
        sketch_estimator_list=sketch_estimator_list,
        scenario_list=scenario_list,
        description_to_file_dir=self.analysis_results[
            KEY_DESCRIPTION_TO_FILE_DIR],
        out_dir=self.out_dir)

    report_html = MESSAGE_HTML_TEMPLATE.format(
        num_estimable_sets_stats_df_html=num_estimable_sets_stats_df_html,
        running_time_df_html=running_time_df_html,
        plot_df_html=plot_df_html)

    return report_html

  
class CardinalityReportGenerator(ReportGenerator):
  """Generate HTML report for the cardinality estimator evaluation."""

  def __init__(self, *args, **kwargs):
    self.error_metric_column = simulator.RELATIVE_ERROR_BASENAME + '1'
    super().__init__(*args, **kwargs)


class FrequencyReportGenerator(ReportGenerator):
  """Generate HTML report for the frequency estimator evaluation."""

  def __init__(self, *args, **kwargs):
    self.error_metric_column = simulator.SHUFFLE_DISTANCE
    super().__init__(*args, **kwargs)

