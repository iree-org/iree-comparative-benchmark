# Common Benchmark Suite

## Onboarding New Models and Benchmarks

To add new benchmarks, there are four steps:

1.  Add the base model implementation
2.  Define model variants on the base model with parameters (e.g. batch sizes)
3.  Define the benchmarks with models and input data
    -   At this point, you should be able to run benchmarks locally with tools.
4.  To publish the benchmarks so they can be run in CI, generate the artifacts
    and upload to GCS, then send out a PR for review.

The figure below shows the files involved in adding new models and benchmarks:

```sh
openxla-benchmark/
├── openxla/benchmark/common_benchmark_suite/
│   ├── models/
│   │   ├── jax
│   │   ├── pt
│   │   │   ├── bert
│   │   │   │   # 1. Add base model implemention
│   │   │   └── "new_base_model"
│   │   │       ├── "new_model.py"
│   │   │       └── "requirements.txt"
│   │   └── tf
│   └── comparative_suite
│       ├── jax
│       ├── pt
│       │   │   # 2. Add "variants" of the model with parameters
│       │   ├── "model_definitions.py"
│       │   │   # 3. Add benchmark cases with models and inputs
│       │   └── "benchamrk_definitions.py"
│       └── tf
└── comparative_benchmark
    ├── jax_xla
    ├── pt_inductor
    │   │   # 3-1. Tools to run benchmarks locally
    │   ├── "setup_env.sh"
    │   └── "run_benchmarks.py"
    └── ...
```

Based on the use cases, step 1 or 2 can be skipped if the definitions already
exist. Check the detailed steps below to see where to find and add definitions.

### 1. Add the base model implementation

To add a new model, it needs to be written with an ML framework (currently only
support JAX/PyTorch/TensorFlow) and the model class need to implement the
interfaces in
[model_interfaces](/common_benchmark_suite/openxla/benchmark/models/model_interfaces.py).

Everything needs to be packed in a Python module and exposes a `create_model`
module method for creating the model object. The `create_model` can take
parameters to initialize the model. For example, the batch size, data type, or
input sequence length.

You can find an example of PyTorch model implementation at
[common_benchmark_suite/openxla/benchmark/models/pt/example](/common_benchmark_suite/openxla/benchmark/models/pt/example).
All model implementations can be found under
[common_benchmark_suite/openxla/benchmark/models](/common_benchmark_suite/openxla/benchmark/models).

### 2. Define variants of models

Once the base model implementation is added, we use Python code to define the
variants of a model with parameters. For example, with a base BERT model
implementation, we can define multiple "variants" of the BERT model to benchmark
with different batch sizes and input length.

Model variants are defined with the class
[def_types.Model](/common_benchmark_suite/openxla/benchmark/def_types.py). An
example model variant `EXAMPLE_FP32_PT_BATCH32` can be written as:

```py
# First describe our base model implementation, including the path to find the
# module of the implementation.
EXAMPLE_PT_IMPL = def_types.ModelImplementation(
    name="EXAMPLE_PT",
    tags=["example"],
    framework_type=def_types.ModelFrameworkType.PYTORCH,
    module_path="openxla.benchmark.models.pt.example.example_model",
    source_info="https://pytorch.org/vision/stable/models/mobilenetv3.html",
)
# Define a model variant on the base model with batch size = 32 and data type = fp32
EXAMPLE_FP32_PT_BATCH32 = def_types.Model(
    name="EXAMPLE_FP32_PT_BATCH32",
    tags=["batch-32", "example"],
    model_impl=EXAMPLE_PT_IMPL,
    model_parameters={
        "batch_size": 32,
        "data_type": "fp32",
    },
)
```

These definitions should be added to `model_definitions.py` in each framework
folder under
[common_benchmark_suite/openxla/benchmark/comparative_suite](/common_benchmark_suite/openxla/benchmark/comparative_suite).

It can be tedious to write variants for each batch size, so we provide utilities
to help generate `def_types.Model` for different batch sizes. See the example
`EXAMPLE_FP32_PT_BATCHES` in
[model_definitions.py](/common_benchmark_suite/openxla/benchmark/comparative_suite/pt/model_definitions.py)
to learn more.

> Remember to add your `def_types.Model` to the list `ALL_MODELS` at the bottom
> of `model_definitions.py`, so tools can find your models.

### 3. Define the benchmarks with models and input data

Once the `def_types.Model` are defined, the benchmark cases
[def_types.BenchmarkCase](/common_benchmark_suite/openxla/benchmark/def_types.py)
to describe the combinations of models and inputs can be added to
`benchmark_definitions.py` in each framework folder under
[common_benchmark_suite/openxla/benchmark/comparative_suite](/common_benchmark_suite/openxla/benchmark/comparative_suite).

An example benchmark case (with the example model `EXAMPLE_FP32_PT_BATCH32`
above) can be written as:

```py
# Define a benchmark case with default input data.
EXAMPLE_FP32_PT_BATCH32_CASE = def_types.BenchmarkCase.build(
  model=EXAMPLE_FP32_PT_BATCH32,
  input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
)
```

> When testdata.INPUT_DATA_MODEL_DEFAULT is specified, the
> `generate_default_inputs` of the model implementation will be called to
> generate test data. This is currently the only supported input data option
> (see #44).

Similar to `def_types.Model`, we also provide utilities to generate
`def_types.BenchmarkCase` for different batch sizes. See the example
`EXAMPLE_FP32_PT_CASES` in
[benchmark_definitions.py](/common_benchmark_suite/openxla/benchmark/comparative_suite/pt/benchmark_definitions.py)
to learn more.

> Remember to add your `def_types.BenchmarkCase` to the list `ALL_BENCHMARKS` at
> the bottom of `benchmark_definitions.py`, so tools can find your benchmark
> cases.

### 3-1. Test your benchmarks locally

At this point, you should be able to run the newly added benchmarks locally.
Depends on which frameworks the benchmarks are based on, you can run benchmark
tools under [comparative_benchmark](/comparative_benchmark) to test them.

> The `benchmark_all.sh` scripts live alongside the benchmark tools usually give
> good examples of how to run benchmarks.

Here is an example to run `EXAMPLE_FP32_PT_BATCH*` models (including all batch
sizes) with PyTorch Inductor on CPU:

```sh
# Ensure your Python version is supported by PyTorch Inductor (e.g. 3.10).
python3 --version

# Setup the virtualenv ./pt-benchmarks.venv and install dependencies.
./comparative_benchmark/pt_inductor/setup_venv.sh

# Activate the virtualenv.
source ./pt-benchmarks.venv/bin/activate

# Run benchmarks on the host CPU. `--generate-artifacts` is required to generate
# input data locally instead of downloading from cache.
./comparative_benchmark/pt_inductor/run_benchmarks.py \
  -name "models/EXAMPLE_FP32_PT_BATCH.+" \
  -device host-cpu \
  -o /tmp/results.json \
  --generate-artifacts \
  --verbose
```

### 4. Submit and publish the new benchmarks

> TODO(pzread): Add instructions to generate and upload test data to GCS cache.
