# Cardinality and Frequency Estimation Evaluation Framework

<br>

## Table of Contents

- [**Context and Objectives**](#context-and-objectives)
- [**What Is a Private Reach & Frequency Estimator?**](#what-is-a-private-reach--frequency-estimator)
- [**Privacy Themes**](#privacy-themes)
- [**Utility and Privacy Trade-off**](#utility-and-privacy-trade-off)
- [**Candidate Cardinality and Frequency Estimators**](#candidate-cardinality-and-frequency-estimators)
  - [Characteristics of Estimators](#characteristics-of-estimators)
- [**Evaluation Plan**](#evaluation-plan)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Reach](#reach)
    - [Frequency](#frequency)
  - [Experiment Setup](#experiment-setup)
    - [Evaluation Variables](#evaluation-variables)
  - [Simulation Scenarios](#simulation-scenarios)
    - [Reach Simulation Scenarios](#reach-simulation-scenarios)
      - [Scenario 1: Independence m-publishers with homogeneous user reach probability](#scenario-1-independence-m-publishers-with-homogeneous-user-reach-probability)
      - [Scenario 2: n-publishers independently serve a remarketing list](#scenario-2-n-publishers-independently-serve-a-remarketing-list)
      - [Scenario 3: m-publishers with heterogeneous users reach probability](#scenario-3-m-publishers-with-heterogeneous-users-reach-probability)
      - [Scenario 4: Full overlap or disjoint](#scenario-4-full-overlap-or-disjoint)
      - [Scenario 5: Sequentially correlated campaigns](#scenario-5-sequentially-correlated-campaigns)
    - [Frequency Simulation Scenarios](#frequency-simulation-scenarios)
      - [Frequency scenario 1: Homogeneous user activities within a publisher](#frequency-scenario-1-homogeneous-user-activities-within-a-publisher)
      - [Frequency scenario 2: Heterogeneous user frequency](#frequency-scenario-2-heterogeneous-user-frequency)
      - [Frequency Scenario 3: Publisher Constant Frequency](#frequency-scenario-3-publisher-constant-frequency)
- [**Next Steps**](#next-steps)
- [**Future Work**](#future-work)
- [**References**](#references)
- [**Appendix 1: Reach Estimates Reporting Plot Examples**](#appendix-1-reach-estimates-reporting-plot-examples)
- [**Appendix 2: Frequency Estimates Reporting Examples**](#appendix-2-frequency-estimates-reporting-examples)
- [**Appendix 3: Example Parameters of Scenarios**](#appendix-3-example-parameters-of-scenarios)

<br>

## Context and Objectives

In this document we present a framework for evaluating privacy-preserving reach and frequency estimation algorithms. Our goal is to identify the best algorithms to constitute the core of the _Private Reach and Frequency Estimator_ component of the Cross-Media Measurement System recently proposed to the WFA.

The Reach and Frequency estimator is downstream of the Virtual People Model in the Cross-Media Measurement System design, where the Virtual People Model provides a unified identity framework as inputs for Reach and Frequency estimators. The framework discussed in this paper is only intended to evaluate the efficacy/accuracy of the proposed reach and frequency estimators. In the future, we plan also to evaluate the end-to-end accuracy of the cross-media measurement system, which depends on the interaction between the Private Reach and Frequency Estimators and Virtual People Model  Despite this, we do believe that such an end-to-end evaluation is both useful and necessary in the long-run. However, we also believe that such tests are best done after the evaluation described herein is complete and the estimators candidates have been finalized.

This document has two main goals. The first is to describe a set of criteria for selecting estimators. This includes both a set of metrics for comparing them as well as a set of privacy themes that describe the environment in which they could be used. The second goal is to prescribe a set of simulation scenarios that test the estimation techniques under consideration. These scenarios are of two types. The first type of scenario tests against use cases typically found in ads measurement, while the second type is intended to stress test the estimators on difficult corner cases.

We welcome any and all input from various constituents to ensure the testing framework meets industry and advertiser needs.

<br>

## What Is a Private Reach & Frequency Estimator?

A private reach and frequency-estimator is a combination of security protocol, a private data representation and an estimation algorithm that would enable a private cross-media measurement solution to combine data from first party measurement data (e.g. publishers’ impression data; advertisers’ conversion data) and estimate the deduplicated reach and frequency. By security protocol, we are referring to the process used for encrypting publisher-provided input data for cross-media measurement. By private data representation, we are referring to how the input data will be presented to enable a private reach and frequency estimation process. By estimation algorithm, we are referring to the process of how deduplicated reach and frequency will be estimated from the input data from different data providers.

Private reach and frequency-estimators are just one component of the cross-media measurement system. Please refer to WFA’s technical blueprint proposal for information about other technical components such as the Virtual People model.

<br>

## Privacy Themes

To understand the level of privacy guarantee the candidate estimators will provide, we designed three privacy themes to guide the evaluation of the proposed algorithms.

The first theme is to consider the accuracy of each estimation technique without any privacy considerations, which practically speaking means with no noise added to the output. While a noiseless method is not a true candidate for adoption, such an evaluation provides a simple baseline that can be compared to the noisy versions in order to determine the impact of noise on accuracy.

The second theme is the local noise theme, which entails adding enough noise to each of the data provider’s input sketches so that they may be seen unencrypted. This means that as sketches are combined the overall noise level will go up and thereby degrade accuracy in estimating reach and frequency. However, one benefit of the local noise approach is that sketches can be easily and freely recombined as advertisers see fit.

The third theme is the global noise theme, and it assumes that publishers will send non-noisy encrypted sketches to a pool of secure multiparty computation workers that will then combine the sketches cryptographically. Once combined, noise can be added just once, which will allow for  better estimation accuracy overall. While this approach will have superior accuracy when compared to the local noise theme, it will demand higher compute costs and latency. To simulate this we will combine unencrypted sketches and add noise after they are combined.

It is worth mentioning that all three themes above are software-based methods. Another  possible method for combining sketches involves hardware-based approaches such as secure enclaves and specifically AMD-SEV or Intel SGX technology. More research is needed in this area to determine both the suitability and availability of these types of systems. This method is out of scope for the purposes of this evaluation and is mentioned here only for the sake of completeness.

Any method that is acceptable for supporting cross-publisher deduplicated reach and frequency estimation in production will necessarily produce noisy output. The question remains how much noise is required and how this impacts utility. This as well as issues surrounding report frequency (e.g. daily, monthly) and multiple reports on the same datasets will be addressed in the next section.

<br>

## Utility and Privacy Trade-off

The evaluation framework proposed herein seeks to understand the privacy-utility tradeoff of the various estimation methods proposed when they are subjected to the privacy themes described in the previous section. The result of this evaluation exercise will be a “report card” for each cardinality estimator, reporting its performance across several dimensions in an effort to facilitate decision making and further discussion in WFA forums.

It is a complex process to decide which private cardinality and frequency estimator to use for cross-media measurement. The various estimators to be evaluated feature a trade-off between not only  privacy and utility, but also operational cost. Indeed, all cardinality estimators considered can provide strong privacy guarantees. However, they may introduce significant operational costs in order to maintain accuracy, or fail to maintain accuracy altogether as the number of publishers is increased. Conversely, all of the estimators may be configured to accurately estimate reach and frequency with low operational cost. However, these configurations might not be able to provide enough privacy guarantees and therefore should be avoided.



A rather complex privacy consideration occurs when generating multiple reports using the same underlying data set. A trivial example of this is just rerunning the same report multiple times, in which case the outputs could be averaged to remove the noise and thereby defeat any privacy protections. That is, as the same underlying data is used to generate multiple reports it is possible to gain information about the structure of the inputs, and therefore it is necessary to add additional noise to the outputs when the same inputs are reused. This of course degrades accuracy.

We see several questions that must be addressed in regard to this issue:


1. Are the privacy budgets independent for non-overlapping daily reports?
2. What is the concern for slices (e.g. demo slices) of a single day’s data? The answer here will depend on whether the slices are disjoint or overlapping.
3. What is the concern for reports that span multiple days in the presence of daily reports?

We believe that the above considerations are important to address, regardless of the estimation method ultimately used. Therefore such considerations will be a topic of future work.

<br>

## Candidate Cardinality and Frequency Estimators

There are three main private cardinality estimator proposals that we are evaluating for WFA-led long-term cross-media solution:

1. **Vector of Counts:** [Vector of Counts](https://docs.google.com/document/d/1iynkte8EnxZxNOLwOFxh_MquKsekvnxlJ3BM-ECKcDI/edit#heading=h.ut20a3rzx6f0) (VoC) is a user count vector of random buckets of user IDs. VoC randomly divides user ids into buckets, and counts the unique users in each bucket. VoC is the aggregated counts that can be filtered to satisfy[ k-anonymity](https://en.wikipedia.org/wiki/K-anonymity) if needed. Noise can also be added in order to make VoCs [differentially private](https://en.wikipedia.org/wiki/Differential_privacy), which is a higher privacy standard used by the government and some data providers (e.g. [Google](https://www.theverge.com/2019/9/5/20850465/google-differential-privacy-open-source-tool-privacy-data-sharing)). The correlation of VoCs from multiple data providers enables an estimate of deduped reach and frequency. Vector of Counts is easy to implement. The pairwise dedupe is unbiased with theoretical bounds. Three or more publishers' dedupe requires approximation to achieve scalability.

2. **Secure Bloom Filters:** Secure Bloom Filters transform the users that are reached by advertising campaigns into multiple Bloom filters. The cross-publisher deduplication is then achieved by combining Bloom filters from multiple publishers via Bloom filter union. Additional privacy protection will be provided through adding noise to the Bloom filters before combination. Secure Bloom filters excel at its low complexity in implementation and its intuitive deduplication. However, if we integrate differential privacy noise with secure Bloom filters for privacy guarantees, it could compound during the combination of the Bloom filters and negatively impact accuracy.

3. **Any Distribution Bloom Filters and Counting Bloom Filters:** Standard uniformly distributed Bloom filters are reasonably accurate cardinality estimators, while Counting Bloom filters provide good accuracy for both cardinality and frequency estimation. However, in both cases the size of the sketch scales linearly with respect to the size of the set to be estimated, which will be problematic for scaling ads measurement.

    To address this issue we have [developed a set of methods](https://research.google/pubs/pub49177/) for applying an arbitrary distribution to the bucket allocation for both counting and standard Bloom filters. These methods reduce the number of buckets required while maintaining accuracy. Moreover, these methods are amenable to both the local privacy theme and the global privacy scheme discussed above.

    We anticipate testing several methods in this family. These includes:

    1. Exponentially distributed Bloom filters
    2. Logarithmically distributed Bloom filters
    3. Counting Bloom filter variants of the two above methods

<br>

### Characteristics of Estimators

The table below summarizes the characteristics of the proposed estimators. These were derived from existing analysis and literature of the algorithms’ scalability in terms of number of publishers and the size of the sketch, as well as other characteristics.


<table>
  <tr>
   <td><strong>Evaluation Criteria</strong>
   </td>
   <td><strong>Secure Bloom Filter with local noise (DP)</strong>
   </td>
   <td><strong>VoC with local noise (DP)</strong>
   </td>
   <td><strong>Any Distribution [Counting] Bloom Filters</strong>
   </td>
  </tr>
  <tr>
   <td>Scalability with increasing # of publishers (in <a href="https://en.wikipedia.org/wiki/Big_O_notation">big-O</a> notation)
   </td>
   <td>O(n)<sup>3</sup>
   </td>
   <td>O(n) (Notable bias for some cases (e.g. ids if reached are only reached  by 2pubs)
   </td>
   <td>O(n)
   </td>
  </tr>
  <tr>
   <td>Scalability of Increasing Abs Set Sizes (in <a href="https://en.wikipedia.org/wiki/Big_O_notation">big-O</a> notation)
   </td>
   <td>O(n)
   </td>
   <td>Constant (4096)<sup>4</sup>
   </td>
   <td>O(log(n))<sup>5</sup>
   </td>
  </tr>
  <tr>
   <td>Private Input
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>Implementation Complexity
   </td>
   <td>Low
   </td>
   <td>Low
   </td>
   <td>High for <a href="https://en.wikipedia.org/wiki/Secure_multi-party_computation">MPC</a>
<p>
Low for Local DP
   </td>
  </tr>
  <tr>
   <td>DP Output
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>Estimating Frequency
   </td>
   <td>Approximation<sup>1</sup>
<p>
   </td>
   <td>Approximation
   </td>
   <td>Yes for MPC-based
<p>
Maybe for non-MPC-based
   </td>
  </tr>
  <tr>
   <td>Share Secret
   </td>
   <td>Yes <sup>2</sup>
   </td>
   <td>Yes
   </td>
   <td>No for MPC-based
<p>
Yes for non-MPC-based
   </td>
  </tr>
  <tr>
   <td>Lossless Union
   </td>
   <td>Yes
   </td>
   <td>No
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>Scalability with increasing cardinality ratio
   </td>
   <td>Yes
   </td>
   <td>TBD
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>Scalability with increasing overlap ratio
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>Accuracy with DP
   </td>
   <td>TBD
   </td>
   <td>TBD
   </td>
   <td>Good for MPC-based
<p>
Non-MPC-based: TBD
   </td>
  </tr>
</table>


**Footnotes:**

1. Higher accuracy with more cohorts.
2. Trusted facilitator and trust no participating publisher will collude with malicious actors.
3. `O(n)`: bucket size required is proportional to _n_, the number of ids in a campaign.
4. `const(4096)`: bucket size is 4096 regardless of the number of ids in a campaign.
5. `O(log(n))`: bucket size is in the order of `log(n)`.
---

<br>

## Evaluation Plan

We outline a test plan below for experimentally evaluating the accuracy of each proposed cardinality and frequency estimator. Experiments consist of running the cardinality estimators on simulated multi-publisher reach data, estimating the cardinality and frequency distribution of the union from multiple publishers’ reach data and reporting the associated accuracy metrics.

As an initial approach, we believe simulated data is sufficient to evaluate the accuracy of the cardinality estimators against various experiments designed to represent more complex scenarios than may exist in real-world campaigns. While we believe this simulated data will help guide initial decisions related to accuracy and privacy trade-offs, we recognize that subsequent phases of testing and evaluation must include real-world campaign data across publishers. To this end, we will evaluate the results of the initial simulated scenarios with consideration of how likely these scenarios are to be present in real-world campaign data, and what the risk of them being present would be.

Currently this framework only considers standalone evaluation of the cardinality and frequency estimation algorithms. Specifically, end-to-end testing with Virtual People[^1] is out of scope for this first round of testing, but we plan to add an end-to-end evaluation that includes Virtual People in the future.


### Evaluation Metrics

The evaluation metrics presented here have been designed for evaluation purposes only, and are meant to be the basis of an evaluation rubric that can help researchers compare the relative performance of the above methods under simulation. Specifically, they do not represent the output metrics of the final cross-media campaign reporting system. That disclaimer out of the way, we will turn to the details of our evaluation metrics.

For each evaluation, we will repeat the data generation and cardinality estimation 100 times. Each repetition is referred to as a run, and an experiment is a group of 100 runs. For each experiment the mean relative error will be reported, and afterwards various statistics can be calculated.


#### Reach

We propose to estimate the performance of reach estimation by computing the relative error for each run. The relative error of reach is defined as:

<p align="center">
<img align="center" src="https://render.githubusercontent.com/render/math?math=Relative\:Error = \frac{(Estimated\:Reach - True\:Reach)}{True\:Reach}">
</p>

The sign of the relative error will help us evaluate the bias of the estimator after many repetitions.

For each evaluation the following will be reported:



*   The number of non-noisy sketches that can be unioned such that 95% of the runs are within a 5% relative error. This corresponds to the first privacy theme. Note that the criteria --95% of runs within 5% relative error -- may be adjusted up or down based upon results, but as of this writing it seems like an achievable target. We chose 95% as the proportion of runs because 95% is a standard measure of confidence; however, the 5% relative error mark was chosen for less principled reasons. It would also be possible to report the number of sketches that can be unioned for multiple such criteria, for example 80/20, in addition to 95/5. Taking this approach would allow the consumers of such noisy sketches to make trade-offs between accuracy and breadth of coverage during ad-hoc analyses. Overall we welcome feedback on these thresholds.
*   A plot of the number of locally differentially private sketches that can be unioned where the union preserves the 95/5 criteria above will be plotted against several values of epsilon (i.e. the noise level). This corresponds to the second privacy theme.
*   Finally, the number of noiseless sketches that can be unioned and have differentially private noise applied during the estimation while preserving the 95/5 criteria will be plotted against several values of epsilon.


#### Frequency

Frequency is a distribution (see example plot in the Appendix) and as such requires different reporting metrics than reach. Our goal here is to evaluate how closely the estimated cross-media frequency distribution is to the truth. For this we propose using _shuffle distance_, which is a metric that measures the difference between two distributions;  it is similar to [edit distance](https://en.wikipedia.org/wiki/Edit_distance), which is a measure of the difference between two strings. Shuffle distance is defined as:

<p align="center">
<img align="center" width="500" height="50" src="https://render.githubusercontent.com/render/math?math=Shuf\!fle\:Distance(freq_1, freq_2) = \frac{1}{2} \times \sum_{k=1}^{freq_{max}} |freq_1[k] - freq_2[k]|">
</p>

Shuffle distance can be interpreted as the minimum fraction of ids that would need to be relabeled in order to achieve a perfect match between two distributions. That is, a shuffle distance of zero indicates a perfect match. On the other hand, a shuffle distance of 30% means that at least 30% of ids would need to be relabeled in order to make the two distributions match.  As an example, consider one frequency distribution: (50%, 30%, 20%), where 50% of the ids have a frequency of one. Another frequency distribution is: (40%, 10%, 50%). The shuffle distance in this case is 30%. That is, we would need to move at least 30% of the ids to achieve a perfect match. Specifically, 10% and 20% would need to be moved from the 1st and the 2nd frequency buckets, respectively, to the 3rd frequency bucket.

For each evaluation the following will be reported:


*   The number of non-noisy sketches that can be unioned such that 80% of the runs are within a 20% shuffle distance. As with the reach reporting metrics, the numbers are somewhat illustrative, and we welcome feedback as to their specific values.
*   For the second privacy theme we will union noisy sketches.
*   For the third privacy theme we will add noise to the union of non-noisy sketches.

These three results will enable us to rank sketches with respect to their performance across our set of privacy themes and choose options that are best suited for each of them. However, note that in general, frequency estimation from noisy sketches is challenging, and in practice, the accuracy requirement will need to be lower than that for reach. For all privacy themes we will also provide a plot that compares the true and estimated frequency distributions with error bars. See the Appendix for an example.


### Experiment Setup

For each individual experiment, we will generate a series of sets <img src="https://render.githubusercontent.com/render/math?math=S_1,S_2,...,S_i"> of cardinality <img src="https://render.githubusercontent.com/render/math?math=c_1,c_2,...,c_i">, where each set represents the campaign reach data from one publisher. The sets will then get transformed into their private sketch representations according to the proposed cardinality estimators. Depending on the interested evaluation variables, the union of the sets, either pairwise or among more than two sets, will be created and estimated for the union cardinality.


### Evaluation Variables

In order to assess the influence of different real-world cross-media campaign factors, we will vary the experiment set up across one or more of the following variables. Note that these variables are high level descriptions of what could be varied and that each of the simulation scenarios below will define precisely how the parameters associated with them can be varied.


#### 1. Sketch Size

This is the size of the sketch representation of the private reach data. Smaller sketch sizes make the measurement more space-efficient and larger sketch sizes will incur more computation cost. The sketch size will be varied as needed for each estimation method.


#### 2. Number of Publishers

Advertisers often run campaigns across more than two publishers and require deduplicated reach reporting across all the media and publishers. To assess the scalability of the cardinality estimator as the number of the publishers increase, we will test the cardinality estimators with up to 100 publishers.


#### 3. Campaign Size

The cross-media cardinality estimator must be accurate for campaigns and publishers of all sizes. To access this scalability requirement, we will test the pairwise union cardinality estimates for campaign sizes of 3k, 30k, 300k, 3M, 30M, 300M. We will also consider increasing the universe size to 1B. These sizes are per-publisher prior to deduplication.


#### 4. Cardinality Ratio

Advertisers often run campaigns across publishers of different sizes;  therefore, estimators must be robust when combining reach data from publishers of all sizes. For this reason the cardinality ratio will be varied in a pairwise scenario, where the cardinality ratio is defined as the cardinality of the larger set’s cardinality divided by the smaller set’s cardinality. The cardinality ratio will be varied from 1 to 10 in increments of 2 and from 10 to 100 in increments of 10.


#### 5. Overlap Ratio

Some cardinality estimators are sensitive to the relative sizes of the overlap between sets and lose accuracy when the overlap is either too small or too large. The size difference of the overlap can be defined as the ratio of the cardinality of the intersection divided by the smallest set among the sets. Therefore, the sizes of the publishers will be varied so that the largest pairwise cardinality ratio for one instance increases from 1 to 100 in increments of 10.


#### 6. Epsilon (Privacy Parameter)

Epsilon is the differential privacy parameter and as such defines the amount of noise that is added to each sketch. Lower values of this parameter result in more noise and thus higher privacy protections. It is important to verify the performance of estimators across a range of values in order to determine the value of epsilon that provides the appropriate utility/privacy-trade-off. To this end we will evaluate against the following values: `(2 ln(3), ln(3), 0.5, 0.25, 0.1)`. We believe that an epsilon of 0.1 will prove to have little utility; however, if it proves useful we will consider additional smaller values of epsilon.

<br>

### Simulation Scenarios

The evaluation variables proposed above discuss the different variables of interest that will impact performance evaluation. This section applies those variables by focusing on simulation scenarios for evaluating methods for deduplicated cross-publisher reach and frequency. In general, we would like to define simulation scenarios that can easily scale to multiple publishers. We first discuss reach simulation scenarios before turning to frequency simulation scenarios in the next section.

Each simulation scenario has parameters that can be varied to support one or more of the evaluation variables described above. For example, [cardinality ratio](#4-cardinality-ratio) can be controlled by varying campaign sizes. We will discuss the parameter settings in the Appendix.

Note that the current simulation scenarios do not have settings that capture potential changes to campaigns while they are in-flight. However, after our first round of evaluation we may design additional simulation scenarios to test such dynamic situations.

<br>

#### Reach Simulation Scenarios

The first three simulation scenarios are motivated by real-world campaigns and situations. The last two simulation scenarios are designed for stress testing the candidate sketch methods.


#### Scenario 1: Independence m-publishers with homogeneous user reach probability

This scenario simulates an advertiser running campaigns on multiple publishers that randomly serve users who go to their websites randomly. This scenario is best modeled by the independence assumption model known as Sainsbury method [1].


#### Scenario 2: n-publishers independently serve a remarketing list

This scenario simulates an advertiser running campaigns on multiple publishers that target a remarketing list. For example, an advertiser collects user ids of users who visit its web site. The advertiser then designs remarketing campaigns running on multiple publishers only serving users in this remarketing list.


#### Scenario 3: m-publishers with heterogeneous users reach probability

Scenario 1 and 2 assume that users have the same reach probability crossing multiple publishers. However, this is rarely the case in real-world cross-media campaigns, where generally users have different reach probabilities across publishers. To address this, scenario 3 models the user’s reach probability [as an exponential distribution, which is a more realistic assumption and is often used to model user activity and reach.](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1ab33ed15e0a7724c7b0b47361da16259fee8eef.pdf)

Furthermore, in cross-media campaigns, a user’s reach probability for one publisher may correlate with the reach probability of another publisher. These correlations often manifest as large overlaps between the reached populations of each publisher. We propose two sub-scenarios to understand the impact of the correlation of users’ reach probabilities when they have heterogeneous reach probabilities.


(a). **[independent user reach probabilities]** the reach probability of a user in a publisher is independent of his/her reach probability in other publishers. For example, this might assume that a user’s reach probability in YouTube is independent of his reach probability in Facebook.

(b). **[fully correlated user reach probabilities]** the reach probability of a user in a publisher  is the same as that in other publishers. For example, this would assume that if a user has the highest level of reach probability in Youtube, he also has the highest level of reach probability in Facebook.


#### Scenario 4: Full overlap or disjoint

This scenario serves as a stress test. Real world campaigns are in between fully disjoint and fully overlapped. We will have higher confidence in an estimator candidate if it works well on these two extreme cases. As a special case,  it is also interesting in stress testing the case that there are some large sets and some small sets that are fully overlapped.


#### Scenario 5: Sequentially correlated campaigns

Sequentially correlated campaigns provide a scalable way to simulate correlated campaigns. It is designed to stress-test the capability of a candidate in handling correlated campaigns. In particular, it generates simulated campaigns in sequence. One campaign is generated to correlate with either all previous generated campaigns or the immediate previous campaign.

<br>

#### Frequency Simulation Scenarios

A user can be reached multiple times in an ad campaign. Cross-publisher frequency deduplication aims to estimate the distribution of the number of ad exposures per user across multiple data providers.

The frequency simulation scenarios to be discussed are built on the top of the reach simulation scenarios. In fact, we generate the reached users per publisher using one of the above reach scenarios before assigning frequencies to each reached user for that publisher.


#### Frequency scenario 1: Homogeneous user activities within a publisher

This scenario assumes that the users within a publisher’s campaign have uniform Poisson frequency distribution. This assumption is made in order to establish a base frequency simulation. The Poisson distribution is a common choice for modeling the number of times an event (ad impression in our case) occurs in an interval of time (the time duration of ad campaigns in our case).  Let <img src="https://render.githubusercontent.com/render/math?math=freq_{i,j}"> represent the frequency of the i-th reached user from the j-th publisher. This can be modeled by a zero-shifted [Poisson model](https://en.wikipedia.org/wiki/Poisson_distribution) with a rate parameter of <img src="https://render.githubusercontent.com/render/math?math=\lambda_j"> . The Poisson distribution is shifted by one to model the reached users as the reached users have at least one impression. Formally,

<p align="center">
<img align="center" src="https://render.githubusercontent.com/render/math?math=freq_{i,j}\sim\:Poisson(\lambda_{j})%2B1">
</p>


<img src="https://render.githubusercontent.com/render/math?math=\lambda_j"> characterizes the average reached user impressions (excluding the first impression) by the j-th publisher.


#### Frequency scenario 2: Heterogeneous user frequency

Users typically have different activities in a website, and thus have different ad exposures. A [Gamma-Poisson model](https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture) is proposed to model the heterogeneity of the ad exposures of users.  This model has been used in the literature. For example, [2] used it to model page views; and [3] used it to model user reach. The model is formally described as

<p align="center">
<img align="center" src="https://render.githubusercontent.com/render/math?math=freq_{i,j}\sim\:Poisson(\lambda_{i,j})%2B1">
</p>

where the Poisson mean (the average frequency of the i-th user at the j-th publisher) <img src="https://render.githubusercontent.com/render/math?math=\lambda_{i,j}"> follow an exponential distribution (or in general, a [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)).  


#### Frequency Scenario 3: Publisher Constant Frequency

In order to stress test the frequency dedupe estimator, we propose the constant frequency simulation scenario:  a publisher serves <img src="https://render.githubusercontent.com/render/math?math=x"> number of impressions to every reached user. This is a challenging scenario for a frequency deduping estimator. In this scenario, the true frequency distribution has positive mass only when frequency being an integral number of the frequency cap. For example, if the frequency is 3, we  should only have reached users with frequency being multiples of 3, such as 3, 6, 9, etc.

---

## Next Steps

Implementation of the infrastructure required to run the evaluations described above has begun and can be found on [github](https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework). It is licensed under the Apache 2.0 license and contributions are welcome. While a complete overview of the design of the evaluation infrastructure itself is out of scope in this document, a major focus of it has been to make it easy to implement and test new estimation methods. Please contact the WFA tech working group for more information, including on how to get involved. We encourage and welcome contributions and constructive criticism.

We expect implementation to proceed as follows:
*   Reach evaluation framework
    *   Simulation harnesses for the three privacy themes
    *   Agreement on scenario settings
    *   Evaluation of methods described herein
*   Frequency evaluation framework
    *   Simulation harness
    *   Agreement on scenario settings
    *   Evaluation of methods described herein
*   Reporting infrastructure
    *   Generation of plots for inclusion in reports
    *   Generation of summary statistics

Once all methods are evaluated a report will be drafted that provides detailed information about the performance of each method under all of the scenarios. We believe that once this is complete it should be possible to identify at least one method for each privacy theme.We hope these results will facilitate the broader WFA cross-media group to make a global recommendation for how best to proceed with the Private Reach and Frequency Estimator component of the overall Cross-Media Measurement System. We are targeting August 2020 (tentative) for completion of the report.


## Future Work
Once the evaluation report is drafted we should have a good understanding of the various estimation methods as they relate to the three privacy themes. However, an important missing piece will be a full end-to-end evaluation with the Virtual ID Framework. Such an evaluation is far from straightforward, and considerable thought will need to be put in to determine how best to approach this task. Given the literature in this area, we are optimistic about the coming evaluation.

Another area of future work is testing the method presented above on real cross-publisher data. Here, too, the difficulty should not be underestimated, as finding a representative data set across publishers that can be shared for such a test will be difficult.

Finally, we anticipate additional methods to be developed and will test those as they become available.


# References
1. [Media strategy and exposure estimation](http://www.labsag.co.uk/demo/Docs/Manuales-de-Simuladores/profesores/Lecturas/Adstrat/media-strategy-and-exposure-estimation.pdf).
2. Danaher, P. Modeling Pageviews across multiple websites with an application to internet reach and frequency prediction. Marketing Science, 26(3), 422-437, 2007.
3. [Measuring Cross-Device Online Audiences](https://research.google/pubs/pub45353/). Google research. Jim Koehler, Evgeny Skvortsov, Sheng Ma, Song Liu, 2016.
4. [Virtual People: Actionable Reach Modeling](https://research.google/pubs/pub48387/). Google research. Evgeny Skvortsov, Jim Koehler, 2019.

---
## Appendix 1: Reach Estimates Reporting Plot Examples


For each parameter setting of each reach scenario, we will provide a box similar to the one below. This plot shows the mean relative error (horizontal line in center), the 50th percentile of the error (boxes), and the 95th percentile of the error (whiskers), as well as outliers (diamonds).

![example boxplot](https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/doc/img/reach_box_plot.png)

<br>

## Appendix 2: Frequency Estimates Reporting Examples

For each parameter setting of each frequency estimation scenario we will provide a plot similar to the one below. The example plots estimated frequency distribution versus the true frequency distribution. The last bucket represents frequency >= 5.

<br>

## Appendix 3: Example Parameters of Scenarios

Each simulation scenario has multiple parameters that we can run for. One option is to run the entire grid of parameters, but it will have many settings. Moreover, based on our simulation goals and intermediate results, we may not need to run all settings. Also, we may randomly sample the parameter space to generate representative settings for evaluation. These issues will be discussed once we define the simulation parameter space.

The following defines the parameter space for each scenario by enumerating the set of all values under consideration. For each scenario, we also provide one example setting of the values and pseudo-code that provides fine detail about how we intend to implement the scenario.


### Terminology and notation in this section

*   Universe size: <img src="https://render.githubusercontent.com/render/math?math=N">. The number of IDs that can be potentially reached by publishers. In what follows, we use publisher and campaign interchangeably to refer to the set of impressions shown by a publisher for a single campaign
*   Number of publishers (or campaigns):<img src="https://render.githubusercontent.com/render/math?math=M">.
*   Reach of each publisher (or campaign):<img src="https://render.githubusercontent.com/render/math?math=\n_{i}">. The number of ID’s reached by the _i_-th publisher (or campaign).

<br>

#### Reach scenario 1: Independent m-publishers

##### One example setting

* <img src="https://render.githubusercontent.com/render/math?math=N=2\times10^5\:(200,000)">
* <img src="https://render.githubusercontent.com/render/math?math=M=100">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2\times10^4 \:(20,000)"> Reach of each publisher is set to 10% (over universe) in this example. This can be varied as needed.

##### More options

First, let’s consider the fixed campaign size. That is, during each simulation, all the campaigns have the same size relative to the given universe size. We test the performance for all the combinations of the following parameters:

*   Universe size: `10 ** 5, 2 * 10 ** 5, 10 ** 6, 2 * 10 ** 6, 3 * 10 ** 6, 5 * 10 ** 6`.
*   Fixed campaign size: 0.1%, 1%, 10%, 20% of the universe size.
*   Number of campaigns: 2, 3, 5, 10, 30, 100.

The reason to do this is to see the trend of the bias/variance of the dedup method, and it may be able to extrapolate the real world situations. US internet population is around  2.5*10**8. Intensive memory and computation are needed to simulate this universe size. Our evaluation will rely on smaller universe sizes, which can be thought as a subsampling from the full universe.  

Next, we consider varying campaigns’ size in the same simulation. That is, the campaigns for deduping have different sizes.

*   Universe size = `2 * 10 ** 6`.
*   Random campaign size from ``{2 * 10 ** 3, 2 * 10 ** 4, 2 * 10 ** 5}``.
*   Number of campaigns = `2, 3, 5, 10, 100`.

Note: randomizing  the campaign sizes across trials may smooth out biases that are associated with pubs having different campaign sizes.

##### **Pseudo-code**

```python
N = 2*10**5
M = 100
n = 2*10**4
campaigns = []
for _ in range(M):
  cur_campaign = np.random.choice(N, n, replace=False)
  campaigns.append(cur_campaign)
```

<br>

#### Reach scenario 2: n-publishers independently serve a remarketing list.


##### One example setting

All the campaign’s reach over the remarketing list is set to either 50% or 75%, respectively; and, as expected, two campaigns will have 25% and 56%[^2] of overlap (over remarketing list) respectively.

* Remarketing list has 20% of universe users: <img src="https://render.githubusercontent.com/render/math?math=remarketingN = 4 \times 10^4">
* <img src="https://render.githubusercontent.com/render/math?math=M=100">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2\times10^4 \: or \: 3 \times 10^4">

##### More options

*   Universe size: `2 * 10 ** 5` (should have no effect for different universe sizes).
*   Remarketing list size: `4 * 10 ** 4, 6 * 10 ** 4, 8 * 10 ** 4, 10 ** 5`.
*   Campaign size:
    *   Fixed in each simulation: 50%, 75% relative to the remarketing list size.
    *   Varying in each simulation: `Uniform(50%, 75%)` relative to the remarketing list size.
*   Number of campaigns: 2, 3, 5, 10, 100.

##### **Pseudo-code**

```python
remarking_N = 4*10**4
M = 100
n = 2*10**4
campaigns = []
for _ in range(M):
  cur_campaign = np.random.choice(remarking_N, n, replace=False)
campaigns.append(cur_campaign)

```

<br>

#### Reach scenario 3a/3b: Heterogeneous reach probabilities

One example:

* <img src="https://render.githubusercontent.com/render/math?math=M=20">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2 \times 10^4">
* Universe Size <img src="https://render.githubusercontent.com/render/math?math=N=10^5">
* `is_independent = True`. A flag whether correlation of user reach distributions are independent (reach scenario 3a) or fully correlated (reach scenario 3b) crossing publishers.

##### Pseudo-code

To model the user reach probability distribution, we use Dirac Mixture with 4 deltas as described in page 14 in [4]. This Dirac Mixture approximates the Exponential Bow model.

```python
N = 2 * 10**5
n = 10**4
reach_rate = n / N
campaigns = []
# Definition of the four deltas in dirac mixture (see page 14 in [4])
alpha = np.array([0.164, 0.388, 0.312, 0.136]) * N
cumsum_alpha = [0] + np.cumsum(alpha)
x = [0.065, 0.4274, 1.275, 3.140]
for _ in range(M):
  campaign = set(np.hstack(
      [np.random.randint(cumsum_alpha[i], cumsum_alpha[i+1],
                         int(reach_rate * x[i] * alpha[i]))
       for i in np.arange(4)]))
  if is_independent:
    # Shuffle ids for a publisher.
    ids = np.arange(N)
    np.random.shuffle(ids)
    campaign = ids[campaign]
  campaigns.append(campaign)

```

<br>

#### Reach scenario 4: Full overlap or full disjoint

##### One example setting
* <img src="https://render.githubusercontent.com/render/math?math=N=2 \times 10^5">
* <img src="https://render.githubusercontent.com/render/math?math=M=100">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2 \times 10^4">

##### Subset case setting

It is also interesting to stress test the case where some large sets and small sets are fully overlapped.  This parameter setting aims for this:

* <img src="https://render.githubusercontent.com/render/math?math=N=2 \times 10^5">
* <img src="https://render.githubusercontent.com/render/math?math=M=100">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=4 \times 10^4">;<img src="https://render.githubusercontent.com/render/math?math=n_3=n_4=...=n_M=2 \times 10^4">


##### More options
*   Universe size: `2 * 10 ** 6, 2.5 * 10 ** 6, 3 * 10 ** 6`.
*   Campaign size:
    *   Fixed in each simulation: 1%, 10% relative to the universe size.
    *   Varying in each simulation: the first campaign has 1% or 10% relative to the universe size, and other campaigns are randomly drawn from the first campaign, with size Uniform(50%, 100%) relative to the first campaign.
*   Number of campaigns: 2, 3, 5, 10, 100.


##### Pseudo-code

```python
N = 2*10**5
M = 100
n = 2*10**4
first_campaign = np.random.choice(N, n, replace=False)
campaigns = [first_campaign] * M
```

<br>

#### Reach scenario 5: Sequentially correlated campaigns

##### One example setting

* <img src="https://render.githubusercontent.com/render/math?math=M=100">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=10^4">
*   `share_prop` = the proportion of IDs of the current campaign to be from the {previous generated campaigns} or {previous campaign} = 0.1 or 0.5.


##### More options



*   Universe size: `3 * 10 ** 8`.
*   Campaign size: `10 ** 4`.
*   Number of campaigns: `100`.
*   Overlap between campaigns:
    *   Fixed: 10%, 50%. That is, for each simulation, the overlap between the current campaign and all the previous ones is fixed.
    *   Varying: Uniform(5%, 15%), Uniform(45%, 55%). That is, for each simulation, the overlap between the current campaign and all the previous ones is randomly drawn from a uniform distribution.


##### Pseudo-code (current campaign is correlated to all previous campaigns):


```python
reach = 10**4
share_prop = 0.1
current_union = np.array([])
campaigns = []
universe = np.arange(some_large_number)
for _ in range(M):
  overlap_size = int(reach * share_prop)
  campaign_ids_overlap = np.random.choice(
    current_union,
    size=overlap_size,
    replace=False)
  campaign_ids_nonoverlap = np.random.choice(
    universe,
    size=n-overlap_size)
  campaigns.append(np.concatenate(campaign_ids_overlap, campaign_ids_nonoverlap))
  current_union = np.union1d(current_union, campaign_ids_nonoverlap)
  universe = np.setdiff1d(universe, campaign_ids_nonoverlap)
```



##### Pseudo-code (current campaign is correlated to the previous campaign):


```python
reach = 10**4
share_prop = 0.1
current_union = np.array([])
campaigns = []
universe = np.arange(some large number)
for _ in range(M):
  overlap_size = int(reach * share_prop)
  campaign_ids_overlap = np.random.choice(
    campaigns[-1],
    size=overlap_size,
    replace=False)
  campaign_ids_nonoverlap = np.random.choice(
    universe,
    size=n-overlap_size)
  campaigns.append(np.concatenate(campaign_ids_overlap, campaign_ids_nonoverlap))
  current_union = np.union1d(current_union, campaigns)
  universe = np.setdiff1d(universe, campaign_ids_nonoverlap)
```

<br>

#### Frequency scenario 1: Homogeneous user activities within a publisher

One example setting

* <img src="https://render.githubusercontent.com/render/math?math=N=2 \times 10^5 (200,000)">
* <img src="https://render.githubusercontent.com/render/math?math=M=100">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2 \times 10^4 (20,000)">
* <img src="https://render.githubusercontent.com/render/math?math=\lambda_1=\lambda_2=...=\lambda_M=1">
*   `Freq cap = False or 3`
*   In this case, reached users of each publisher have an average frequency of 2.

<br>


#### Frequency scenario 2: Heterogeneous user frequency

One example setting
* <img src="https://render.githubusercontent.com/render/math?math=N=2 \times 10^5 (200,000)">
* <img src="https://render.githubusercontent.com/render/math?math=M=10">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2 \times 10^4 (20,000)">
*   `Freq cap = False or 3`
*   At each pub, each reached user has a <img src="https://render.githubusercontent.com/render/math?math=\lambda_i"> independently drawn from the exponential distribution distribution with rate = 1. Then the frequency of each reached user is independently drawn from <img src="https://render.githubusercontent.com/render/math?math=Poisson(\lambda_i)%2B1">.

<br>


#### Frequency scenario 3: Per-publisher frequency capping

The following is a stress testing, in which each publisher serves 3 impressions to every reached id.

One example setting

* <img src="https://render.githubusercontent.com/render/math?math=N=2 \times 10^5 \: (200,000)">
* <img src="https://render.githubusercontent.com/render/math?math=M=10">
* <img src="https://render.githubusercontent.com/render/math?math=n_1=n_2=...=n_M=2 \times 10^4 \: (20,000)">
*   At each pub, each reached user has a frequency = 3.
