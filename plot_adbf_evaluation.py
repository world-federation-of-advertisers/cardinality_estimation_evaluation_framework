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

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from wfa_cardinality_estimation_evaluation_framework.common import plotting
from wfa_cardinality_estimation_evaluation_framework.evaluations import evaluator
from wfa_cardinality_estimation_evaluation_framework.simulations import simulator
from wfa_cardinality_estimation_evaluation_framework.evaluations import analyzer


## simulation 1
# Get all the raw results.
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="1_vary_flip_prob")
raw_df = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))
raw_df["flipping probabaility"] = \
    raw_df["sketch_estimator"].str.replace(".*_", "", regex=True)
raw_df["bloom filter"] = pd.Categorical(
    raw_df["sketch_estimator"].str.replace("_.*", "", regex=True), 
    categories=["exp", "log", "geo"], ordered=False)

df = raw_df.query('num_sets == 10')
# print(df)
plt.figure(figsize=(6,4))
plt.hlines(0, -1, 4, colors="grey", linestyles="dashed")
sns.boxplot(
    x="flipping probabaility", y="relative_error_1", 
    hue="bloom filter", data=df)
plt.title(f"relative error vs flip prob (k=3, n=5000, m=10000)")
plt.ylabel("relative error (100 trials)")
plt.savefig('1_re_vs_p.pdf')
plt.close()


## simulation 2
# Get all the raw results.
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="2_vary_set_size")
raw_df = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))
# print(raw_df.loc[raw_df["relative_error_1"] > 2, ])
# raw_df.loc["relative_error_1"]
raw_df["scenario"] = pd.Categorical(
    raw_df["scenario"], 
    categories=["0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "32.0", "64.0"])
raw_df["bloom filter"] = pd.Categorical(
    raw_df["sketch_estimator"].str.replace("_.*", "", regex=True), 
    categories=["exp", "log", "geo"], ordered=False)
df = raw_df.query('num_sets == 3')
# print(df)
plt.figure(figsize=(8,4))
plt.hlines(0, -1, 8, colors="grey", linestyles="dashed")
sns.boxplot(
    x="scenario", y="relative_error_1", 
    hue="bloom filter", data=df)
plt.title(f"relative error (k=3, m=10000, p=0.15)")
plt.ylabel("relative error (50 trials)")
plt.xlabel("set size / sketch size (ratio)")
plt.savefig('2_re_vs_ratio.pdf')
plt.close()


## simulation 3
# Get all the raw results.
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="3_vary_decay_rate_10k")
raw_df = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))
# print(raw_df.loc[raw_df["relative_error_1"] > 2, ])
# raw_df.loc["relative_error_1"]
raw_df["scenario"] = pd.Categorical(
    raw_df["scenario"], 
    categories=["4", "8", "16", "32", "64"])
raw_df["decay rate"] = pd.Categorical(
    raw_df["sketch_estimator"].str.replace(".*_", "", regex=True), 
    categories=["5", "10", "15", "20"], ordered=False)
df = raw_df.query('num_sets == 3')
# print(df)
plt.figure(figsize=(7,4))
plt.hlines(0, -1, 5, colors="grey", linestyles="dashed")
sns.boxplot(
    x="scenario", y="relative_error_1", hue="decay rate", data=df)
plt.title(f"relative error by exp BF (k=3, m=10000, p=0.15)")
plt.ylim(-1.2, 1.2)
plt.ylabel("relative error (50 trials)")
plt.xlabel("set size / sketch size (ratio)")
plt.legend(loc='upper left')
plt.savefig('3_re_vs_decay_rate_1.pdf')
plt.close()

## Get all the raw results.
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="3_vary_decay_rate_1k")
raw_df_1k = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))
raw_df_1k["sketch size"] = "1000"
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="3_vary_decay_rate_10k")
raw_df_10k = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))
raw_df_10k["sketch size"] = "10000"
evaluation_file_dirs = evaluator.load_directory_tree(
    out_dir=".",
    run_name="eval_adbf_result",
    evaluation_name="3_vary_decay_rate_100k")
raw_df_100k = (
    analyzer.CardinalityEstimatorEvaluationAnalyzer
    .read_evaluation_results(evaluation_file_dirs))
raw_df_100k["sketch size"] = "100000"
raw_df = pd.concat([raw_df_1k, raw_df_10k], axis=0, ignore_index=True)
raw_df["scenario"] = pd.Categorical(
    raw_df["scenario"], 
    categories=["4", "8", "16", "32", "64"])
raw_df["decay rate"] = pd.Categorical(
    raw_df["sketch_estimator"].str.replace(".*_", "", regex=True), 
    categories=["5", "10", "15", "20"], ordered=False)
df = raw_df.query('num_sets == 3 or num_sets == 5')
plt.figure(figsize=(7,4))
plt.hlines(0, -1, 5, colors="grey", linestyles="dashed")
g = sns.catplot(
    x="scenario", y="relative_error_1", hue="decay rate", 
    col="sketch size", row="num_sets", kind="box", data=df)
(g.set_axis_labels(
    "set size / sketch size (ratio)", "relative error (50 trials)")
  .set(ylim=(-1.5, 1.5)))
plt.savefig('3_re_vs_decay_rate_2.pdf')
plt.close()
