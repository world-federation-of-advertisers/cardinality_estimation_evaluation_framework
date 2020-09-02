# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from wfa_cardinality_estimation_evaluation_framework.common import plotting
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator
from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer


# Get all the raw results.
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="4_various")
raw_df = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))

raw_df.to_csv("raw_df.csv", index=False)

df = raw_df.groupby(["num_sets", "sketch_estimator", "scenario"])\
    .agg({'relative_error_1': ['mean', 'std']})
df.columns = ['re_mean', 're_std']
df = df.reset_index()
# df["re_std_sqrt"] = np.sqrt(df["re_std"])
df["re_std_sqrt_inv"] = 1 / np.sqrt(df["re_std"])
df["re_std_log"] = np.log(df["re_std"])
df["re_std_log_cens"] = df["re_std_log"]
df.loc[df["re_std_log_cens"] > 0, "re_std_log_cens"] = 0
df["universe_size"] = (1000000 * df["scenario"].astype(float)).astype(int)
df["log10_universe_size"] = np.log10(df["universe_size"])
df["sketch_size"] = (
    1000 * df["sketch_estimator"]
    .str.replace("k_.*", "", regex=True)
    .astype(int)
)
df["log10_sketch_size"] = np.log10(df["sketch_size"])
df["flip_prob"] = (
    df["sketch_estimator"]
    .str.replace(".*_", "", regex=True)
    .astype(float)
)
df = df.query('num_sets % 5 == 0')
df["num_sets"] = df["num_sets"].astype(int)
# df = df.drop(["sketch_estimator", "scenario"], axis=1)
print(df)

## pairs
sns.pairplot(
    df, 
    vars=["re_std_log_cens", 
          "num_sets", "log10_universe_size", 
          "log10_sketch_size", "flip_prob"])
plt.savefig('4_various_1.pdf')
plt.close()

## raw re vs prob
sns.catplot(
    x="flip_prob", y="re_std", hue="sketch_size", 
    col="num_sets", col_wrap=2, kind="box", data=df)
plt.savefig('4_various_2.pdf')
plt.close()

## transformed re vs prob
sns.catplot(
    x="flip_prob", y="re_std_log_cens", palette="GnBu",
    kind="box", data=df)
plt.savefig('4_various_3(1).pdf')
plt.close()
sns.catplot(
    x="flip_prob", y="re_std_log_cens", hue="sketch_size", 
    kind="box", data=df)
plt.savefig('4_various_3(2).pdf')
plt.close()
sns.catplot(
    x="flip_prob", y="re_std_log_cens", hue="sketch_size", 
    col="num_sets", col_wrap=2, kind="box", data=df)
plt.savefig('4_various_3.pdf')
plt.close()

## re vs prob at k=10
df1 = df.query('num_sets == 10')
sns.catplot(
    x="flip_prob", y="re_std_log_cens", hue="sketch_size", 
    col="num_sets", kind="box", data=df1)
plt.savefig('4_various_4.pdf')
plt.close()

## transformed re vs universe size
sns.swarmplot(x="universe_size", y="re_std_log_cens", data=df)
plt.savefig('4_various_5.pdf')
plt.close()

## regression (https://www.statsmodels.org/devel/examples/notebooks/generated/regression_diagnostics.html)
## 1st-order
y = df["re_std_log_cens"]
X = df[['flip_prob', 'log10_sketch_size', 'num_sets', 'log10_universe_size']]
X = sm.add_constant(X) 
res = sm.OLS(y, X).fit()
print(res.summary())

# (1) remove universe size,
# (2) add interaction term 
# result: AIC/BIC get decreased, which is preferred
X['p_log10_m'] = X['log10_sketch_size'] * np.sqrt(X['flip_prob'])
X = X.drop(["log10_universe_size"], axis=1)
res = sm.OLS(y, X).fit()
print(res.summary())

# ## residual plot
# residuals = res.resid
# student_residuals = res.get_influence().resid_studentized_internal
# sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
# sqrt_student_residuals.index = res.resid.index
# fitted = res.fittedvalues
# smoothed = lowess(sqrt_student_residuals,fitted)
# top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]

# fig, ax = plt.subplots()
# ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
# ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
# ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
# ax.set_xlabel('Fitted Values')
# ax.set_title('Scale-Location')
# ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
# for i in top3.index:
#     ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
# plt.show()


# Get all the raw results.
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="5_prediction")
raw_df = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))

raw_df.to_csv("raw_df.csv", index=False)

df = raw_df.groupby(["num_sets", "sketch_estimator", "scenario"])\
    .agg({'relative_error_1': ['mean', 'std']})
df.columns = ['re_mean', 're_std']
df = df.reset_index()
df["sketch_size"] = (
    1000 * df["sketch_estimator"]
    .str.replace("k_.*", "", regex=True)
    .astype(int)
)
df["flip_prob"] = (
    df["sketch_estimator"]
    .str.replace(".*_", "", regex=True)
    .astype(float)
)
df = df.query('num_sets % 5 == 0 and scenario == "10.0" and sketch_size == 100000')
print(df)
