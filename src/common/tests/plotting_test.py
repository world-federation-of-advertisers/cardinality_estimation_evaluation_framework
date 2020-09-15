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

"""Tests for wfa_cardinality_estimation_evaluation_framework.common.tests.plotting."""

from absl.testing import absltest
import matplotlib
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.common import plotting


def get_relative_error_by_num_sets():
  return pd.DataFrame({
      'num_sets': [1, 1, 2, 1, 2, 2],
      'relative_error': [0.05, -0.05, 0.1, 0.1, -0.1, -0.2]})


def get_df_for_test_barplot_frequency_distributions():
  return pd.DataFrame({
      'run_index': [0, 0, 0, 0, 1, 1, 1, 1],
      'cardinality_source': ['estimated_cardinality', 'estimated_cardinality',
                             'true_cardinality', 'true_cardinality'] * 2,
      'frequency_level': [1, 2, 1, 2] * 2,
      'cardinality_value': [12, 5, 10, 7, 8, 8, 10, 7]
  })


class PlottingTest(absltest.TestCase):

  def test_boxplot_relative_error_plot_xticks(self):
    df = get_relative_error_by_num_sets()
    ax = plotting.boxplot_relative_error(
        df, 'num_sets', 'relative_error')
    xlabels = [x.get_text() for x in ax.get_xticklabels()]
    expected = [str(x) for x in sorted(df['num_sets'].unique())]
    self.assertListEqual(xlabels, expected)

  def test_boxplot_relative_error_plot_returns(self):
    ax = plotting.boxplot_relative_error(
        get_relative_error_by_num_sets(),
        'num_sets', 'relative_error')
    self.assertIsInstance(ax, matplotlib.axes.Axes)

  def test_boxplot_relative_error_plot_raise_column_not_found(self):
    msg = 'num_sets or relative_error not found in df.'
    with self.assertRaisesRegex(ValueError, msg):
      _ = plotting.boxplot_relative_error(
          get_relative_error_by_num_sets(),
          'num_set', 'relative_error')
    with self.assertRaisesRegex(ValueError, msg):
      _ = plotting.boxplot_relative_error(
          get_relative_error_by_num_sets(),
          'num_set', 'rel_err')

  def test_barplot_frequency_distributions_labels_correct(self):
    df = get_df_for_test_barplot_frequency_distributions()
    ax = plotting.barplot_frequency_distributions(
        df=df,
        frequency='frequency_level',
        cardinality='cardinality_value',
        source='cardinality_source')
    xlabels = [x.get_text() for x in ax.get_xticklabels()]
    self.assertListEqual(xlabels, ['1', '2'],
                         'The x-axis tick labels are not correct.')
    self.assertEqual(ax.get_xlabel(), 'Per frequency level',
                     'The x-axis label is not correct')
    self.assertEqual(ax.get_ylabel(), 'Cardinality',
                     'The y-axis label is not correct')

  def test_barplot_frequency_distributions_returns(self):
    df = get_df_for_test_barplot_frequency_distributions()
    ax = plotting.barplot_frequency_distributions(
        df=df,
        frequency='frequency_level',
        cardinality='cardinality_value',
        source='cardinality_source')
    self.assertIsInstance(ax, matplotlib.axes.Axes)


if __name__ == '__main__':
  absltest.main()
