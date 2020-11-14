# Evaluation Framework Development Workflow

<br>

## Table of Contents

- [**Objectives**](#objectives)
- [**Add New Estimators/Data/etc**](#add-new-estimators-data-etc)
    - [**Sketches, Estimators and Local Noisers**](#sketches-estimators-local-noisers)
    - [**Simulation Data Generator**](#simulation-data-generator)
    - [**Global Noisers**](#global-noisers)
- [**Add New Configurations**](#add-new-configurations)
    - [**SketchEstimatorConfig**](#sketch-estimator-config)
    - [**EvaluationConfig**](#evaluation-config)
    - [**Unit testing**](#unit-testing)

<br>

## Objectives

In this document, we describe the modules of the evaluation framework, how to add new sketches, estimators, noisers, simulation scenarios, and configurations to the framework, and run unit testings to make sure that the new components work smoothly.

<br>

## Add New Estimators/Data/etc

### Sketches, Estimators and Local Noisers

All the sketches and estimators are defined in the folder [src/estimators](../src/estimators). All sketches and estimators defined in the file should inherit from the `SketchBase` and `EstimatorBase` defined in [base.py](../src/estimators/base.py) respectively. An example can be found in the [src/estimators/exact_set.py](../src/estimators/exact_set.py). The `ExactMultiSet` defines a sketch supports instance methods including `add` (or `add_ids`). The sketch constructor (`__init__`) should take a `random_seed` for reproducibility.

A certain sketch type may have multiple estimators, and they should be defined in the same file if possible. For instance, in the [src/estimators/exact_set.py](../src/estimators/exact_set.py) file, `LosslessEstimator` and `LessOneEstimator` are two estimators for a list of `ExactMultiSet` sketches.

In the local DP theme, sketches are noised before being sent to the estimator. A noiser instance takes a sketch and returns a (copy of) noised sketch. Any new local noisers should conform to `SketchNoiserBase` defined in [src/estimators/base.py](../src/estimators/base.py). The constructor should take a `random_state` for reproducibility.


### Simulation Data Generator

All the simulation data generators are defined in [src/simulations/set_generator.py](../src/simulations/set_generator.py) and [src/simulations/frequency_set_generator.py](../src/simulations/frequency_set_generator.py) for reach and frequency evaluation respectively, which should inherit from the `SetGeneratorBase` in the [src/simulations/set_generator_base.py](../src/simulations/set_generator_base.py).

### Global Noisers

In MPC, one variant of a global DP noiser takes a merged sketch and adds noises to it. In our framework, the feature of a global DP noiser can be added in the estimator. An example is the `FirstMomentEstimator` defined in [src/estimators/bloom_filters.py](../src/estimators/bloom_filters.py). It has an argument `noiser` which takes the sum of active registers of the merged Any Distribution Bloom Filter, and returns a noised sum.

<br>

## Add New Configurations

Once the developers add new estimators, noisers, estimators, or set generators, they may want to test the new components. Some developers may also want to try out new parameters for the existing or new components. This evaluation framework can run evaluations, which consists of one or more simulations with different sketches, noisers, estimators and set generators, analyzes the results, and generates an HTML report. To run the evaluation, analysis and to generate report, developers need to use the existing or write new configurations in [src/evaluations/data/evaluation_configs.py](../src/evaluations/data/evaluation_configs.py).

### SketchEstimatorConfig

To add the new configurations of estimators, developers need to add `SketchEstimatorConfig` to the returned object of `_generate_cardinality_estimator_configs` for the cardinality estimators or `_generate_frequency_estimator_configs` for the frequency estimators in [src/evaluations/data/evaluation_configs.py](../src/evaluations/data/evaluation_configs.py). The `SketchEstimatorConfig` combines the sketch, noiser, local DP noiser, final estimate noiser (possibly to mimic the global DP noiser), and maximum frequency level to estimate flexibly. In the following example, we define a Bloom Filter of length equal to 1000000 estimator with a global DP noiser ($\epsilon = log(3)$):
```
SketchEstimatorConfig(
    name=construct_sketch_estimator_config_name(
        sketch_name='bloom_filter',
        sketch_config='length_1000000',
        estimator_name='union_estimator',
        estimate_epsilon='log(3)'),
    sketch_factory=bloom_filters.UniformBloomFilter.get_sketch_factory(
        1000000),
    estimator=bloom_filters.FirstMomentEstimator(
        method=bloom_filters.FirstMomentEstimator.METHOD_UNIFORM,
        noiser=estimator_noisers.GeometricEstimateNoiser(
            epsilon=math.log(3))),
    sketch_noiser=sketch_noiser,
)
```
Note that in the above example, we construct `name` attribute of the `SketchEstimatorConfig` by the `construct_sketch_estimator_config_name` method defined in [src/evaluations/data/evaluation_configs.py](../src/evaluations/data/evaluation_configs.py). As the `name` will be parsed in the `analyzer` and the `report_generator` module, it is recommended to use this method so as to avoid bugs.

### EvaluationConfig

To add new evaluation configurations, a developer needs to do add a callable in the returned object in the `_generate_evaluation_configs`. The callable takes two arguments `num_runs` and `universe_size`, and returns an `EvaluationConfig`. An `EvaluationConfig` defines the number of runs and a set of simulation scenarios, aka `ScenarioConfig` which consists of a `name` argument and a `set_generator_factory`. For example, the `_smoke_test` returns an `EvaluationConfig` that contains five simulation scenarios.

### Unit testing

Every new `SketchEstimatorConfig` will be tested in the [tests/interoperability_test.py](../tests/interoperability_test.py) to check if it can be run through with smoke tests. Yet new `EvaluationConfig`s needs to be tested additionally, as it may contain lots of new scenarios and it would be slow to run through.
