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

"""Plot utilities for visualizing cardinality estimation and errors."""

import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_relative_error(df, num_sets, relative_error,
                           metric_name='Relative error'):
  """Boxplot for relative error by number of sets.

  Args:
    df: a pd.DataFrame that has columns of the number of sets and the relative
      error from one or more runs, specified by num_sets and relative_error.
    num_sets: a column name in df specifying the number of sets.
    relative_error: a column name in df specifying the relative_error.
    metric_name: Name of metric to be displayed on y axis.

  Returns:
    A matplotlib.axes.Axes object of boxplot.
  """
  if not set([num_sets, relative_error]).issubset(df.columns):
    raise ValueError('num_sets or relative_error not found in df.')
  _, ax = plt.subplots()
  sns.boxplot(x=num_sets, y=relative_error, data=df, ax=ax)
  ax.plot(ax.get_xlim(), (0, 0), '--m')
  ax.set_ylabel(metric_name)
  ax.set_xlabel('Number of sets')
  return ax


def barplot_frequency_distributions(df, frequency, cardinality, source):
  """Barplot for comparing multiple frequency distributions.

  Args:
    df: a pd.DataFrame that has the columns of the frequency level, the
      cardinality, and the source, specified by frequency, cardinality, and
      source respectively.
    frequency: a column name in df specifying the per frequency level.
    cardinality: a column name in df specifying the cardinality of the set of
      the corresponding frequency level.
    source: a column name in df specifying the source of the cardinality.
      For example, it could be a column containing labels like
      'estimated_cardinality' and 'true_cardinality'. Different source will
      have different colors in the plot.

  Returns:
    A matplotlib.axes.Axes object of the barplot.
  """
  _, ax = plt.subplots()
  sns.catplot(x=frequency, y=cardinality, hue=source, data=df,
              kind='bar', palette='muted', ax=ax)
  ax.set(xlabel='Per frequency level', ylabel='Cardinality')
  return ax
