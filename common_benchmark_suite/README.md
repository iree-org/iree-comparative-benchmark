# Common Benchmark Suite

## Onboarding New Models and Benchmarks

There are four steps to add new models and benchmarks:

1. Add the base model implementation.
2. Define model variants on the base model with parameters (e.g. batch sizes).
3. Define the benchmarks with models and input data.
    - At this point, you should be able to run benchmarks locally with tools.
4. To publish the benchmarks and run in CI, generate and upload the artifacts to
   GCS, then send out a PR for review.

Based on the use cases, step 1 or 2 can be skipped if the definitions already
exist. See [the detailed steps](#step-1-add-the-base-model-implementation) below
for where to find and add definitions.

> **In the example** we use a PyTorch model to demonstrate the process.
> Nevertheless, JAX and TensorFlow models are also supported and you can find
> the corresponding directories/files beside the PyTorch ones.

You can also check these commits to learn how an example benchmark is added:

1. Add the base model implementation:
    [8bd49ab](https://github.com/openxla/openxla-benchmark/commit/8bd49ab90f3b2ae3a399aa444cb091b72fdd82e8)
2. Define model variants:
    [8cc5b67](https://github.com/openxla/openxla-benchmark/commit/8cc5b6712e8e7f3746c7af6fe9d2371f59814d08)
3. Define benchmark cases:
    [f6b295c](https://github.com/openxla/openxla-benchmark/commit/f6b295c5182c0e32c3c67eb48c440c4ffdd2012d)
4. Upload and update artifact urls:
    [41f2e9b](https://github.com/openxla/openxla-benchmark/commit/41f2e9bb1c3598daba58ca5e7015372f4065114c)

To help navigate through the steps, the figure below shows the files involved in
adding new models and benchmarks:

```sh
openxla-benchmark/
├── openxla/benchmark/common_benchmark_suite/
│   ├── models/
│   │   ├── jax
│   │   ├── pt
│   │   │   ├── bert
│   │   │   │   # Step 1. Add the base model implemention.
│   │   │   └── "new_base_model"
│   │   │       ├── "new_base_model.py"
│   │   │       ├── "requirements.txt"
│   │   │       └── "__init__.py"
│   │   └── tf
│   └── comparative_suite
│       ├── jax
│       ├── pt
│       │   │   # Step 2. Add "variants" of the model with parameters.
│       │   ├── "model_definitions.py"
│       │   │   # Step 3. Add benchmark cases with models and inputs.
│       │   └── "benchmark_definitions.py"
│       └── tf
└── comparative_benchmark
    ├── jax_xla
    ├── pt_inductor
    │   │   # Step 3-1. Tools to run benchmarks locally.
    │   ├── "setup_env.sh"
    │   └── "run_benchmarks.py"
    └── ...
```

### Step 1. Add the base model implementation

To add a new model, you need to write the model class with a supported ML
framework (currently support JAX, PyTorch, and TensorFlow) and implement the
[model_interfaces](/common_benchmark_suite/openxla/benchmark/models/model_interfaces.py).
Later this model class will be referenced in benchmarks so tools know how to
create the model.

A simple base model implementation can look like:

```py
# openxla/benchmark/common_benchmark_suite/models/new_base_model/new_base_model.py

import transfomer
from openxla.benchmark.models import model_interfaces

class NewBaseModel(model_interfaces.InferenceModel):
  def __init__(self, batch_size: int):
    self.batch_size = batch_size
    self.model = transfomer.Model(...)
    self.tokenizer = transfomer.Tokenizer(...)

 def generate_default_inputs(self):
   return "The quick brown fox jumps over a lazy dog"

 def preprocess(self, input_text):
   return self.tokenizer(input_text).repeat(self.batch_size)

 def forward(self, batch_tokenized_input):
   return self.model.forward(batch_tokenized_input)

def create_model(batch_size: int):
  return NewBaseModel(batch_size)
```

The code should be located at:

```sh
openxla/benchmark/common_benchmark_suite/models/${FRAMEWORK}/${BASE_MODEL_NAME}/${BASE_MODEL_NAME}.py
```

An optional `requirements.txt` can be created together for the needed packages
during benchmarking.

The model class takes parameters and initializes a model object that can be run
by the corresponding ML framework. It can simply load a model from the other
framework, e.g., Hugging Face, and forward function calls to the actual model.

In addition to the model class, a public method `create_model` needs to be
exposed from the module and returns an instance of the model class. It can take
parameters to initialize the model, for example, the batch size and data type.

You can find an example of PyTorch model implementation at
[example_model.py](/common_benchmark_suite/openxla/benchmark/models/pt/example/example_model.py).

All model implementations can be found under
[common_benchmark_suite/openxla/benchmark/models](/common_benchmark_suite/openxla/benchmark/models).

### Step 2. Define variants of models

Once the base model implementation is added, we now write Python code to define
the variants of a base model with parameters. For example, a base BERT model
implementation can have multiple "variants" to benchmark with different batch
sizes and input lengths.

Model variants are defined with the class
[def_types.Model](/common_benchmark_suite/openxla/benchmark/def_types.py). Note
that its `model_parameters` field will be passed as keywaord arguments to
`create_model` of the base model implementation, so it can control how the base
model is initialized.

An example model variant `EXAMPLE_FP32_PT_BATCH32` can be written as:

```py
from openxla.benchmark import def_types

# First describe our base model implementation, including the path to find the
# module of the code.
EXAMPLE_PT_IMPL = def_types.ModelImplementation(
    name="EXAMPLE_PT",
    tags=["example"],
    framework_type=def_types.ModelFrameworkType.PYTORCH,
    # This path will be loaded with importlib.import_module.
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
[comparative_suite](/common_benchmark_suite/openxla/benchmark/comparative_suite).

It can be repetitive to write variants for each batch size, so we provide
templating helpers to generate `def_types.Model` for different batch sizes. See
the example `EXAMPLE_FP32_PT_BATCHES` in
[model_definitions.py](/common_benchmark_suite/openxla/benchmark/comparative_suite/pt/model_definitions.py)
to learn more.

Please ignore the optional field `artifacts_dir_url` and `exported_model_types`,
they are not required and will be explained in
[the step 4](#step-4-submit-and-publish-the-new-benchmarks).

> **Remember** to add your `def_types.Model` to the list `ALL_MODELS` at the
> bottom of `model_definitions.py`, so tools can find your models.

### Step 3. Define the benchmarks with models and input data

After the `def_types.Model` are defined, the benchmark cases
[def_types.BenchmarkCase](/common_benchmark_suite/openxla/benchmark/def_types.py),
which describe the combinations of models and inputs, can be added to
`benchmark_definitions.py` in each framework folder under
[comparative_suite](/common_benchmark_suite/openxla/benchmark/comparative_suite).

An example benchmark case (with the example model `EXAMPLE_FP32_PT_BATCH32`
above) can be written as:

```py
from openxla.benchmark import def_types, testdata

# Define a benchmark case with default input data.
EXAMPLE_FP32_PT_BATCH32_CASE = def_types.BenchmarkCase.build(
  model=EXAMPLE_FP32_PT_BATCH32,
  input_data=testdata.INPUT_DATA_MODEL_DEFAULT,
)
```

> When testdata.INPUT_DATA_MODEL_DEFAULT is used, the `generate_default_inputs`
> in the model implementation will be called to get test data. This is currently
> the only supported input data option (see #44).

Similar to `def_types.Model`, we also provide templating helpers to generate
`def_types.BenchmarkCase` for different batch sizes. See the example
`EXAMPLE_FP32_PT_CASES` in
[benchmark_definitions.py](/common_benchmark_suite/openxla/benchmark/comparative_suite/pt/benchmark_definitions.py)
to learn more.

> **Remember** to add your `def_types.BenchmarkCase` to the list
> `ALL_BENCHMARKS` at the bottom of `benchmark_definitions.py`, so tools can
> find your benchmark cases.

### Step 3-1. Test your benchmarks locally

At this point, you should be able to run the added benchmarks locally. Depending
on which frameworks the benchmarks are based on, you can run benchmark tools
under [comparative_benchmark](/comparative_benchmark) to test them.

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

### Step 4. Submit and publish the new benchmarks

To submit your benchmarks and potentially run in CI nightly, some artifacts need
to be generated and uploaded to GCS.

> **Before starting**, this step requires permissions to upload files to
> `gs://iree-model-artifacts`. If you don't have access, please skip this step
> and send a PR directly. The core team members can help generate and upload
> artifacts.

There are two kinds of artifacts that will be generated in this step:

1. Preprocessed input data and reference outputs.
2. Exported models (e.g. MLIR) for compiler-level benchmarks (e.g. XLA-HLO,
    IREE).

Depends on which frameworks the benchmarks use, you can run
`scripts/generate_model_artifacts.sh` in each framework folder under
[comparative_suite](/common_benchmark_suite/openxla/benchmark/comparative_suite)
to generate artifacts. Note that sometimes it can be tricky to generate exported
model artifacts. Please read the instructions in the scripts or file issues if
you run into some problems.

The example below generates artifacts for `EXAMPLE_FP32_PT_BATCH*` models:

```sh
./common_benchmark_suite/openxla/benchmark/comparative_suite/pt/scripts/generate_model_artifacts.sh 'EXAMPLE_FP32_PT_BATCH.+'
```

After the artifacts are generated, by default they will be stored at:

```sh
/tmp/${ARTIFACTS_VERSION}/
# where ${ARTIFACTS_VERSION} follows the format "${FRAMEWORK}_models_${FRAMEWORK_VERISON}_$(date +'%s')"

# For example:
# /tmp/pt_models_20230723.908_1690223270/
```

Now upload the aritfacts to GCS by:

```sh
gcloud storage cp -r  "/tmp/${ARTIFACTS_VERSION}" "gs://iree-model-artifacts/${FULL_QUALIFIED_FRAMEWORK}"

# "${FULL_QUALIFIED_FRAMEWORK}" currently can be:
# - jax
# - pytorch
# - tensorflow

# For example:
# gcloud storage cp -r /tmp/pt_models_20230723.908_1690223270 gs://iree-model-artifacts/pytorch
```

Once the artifacts are in GCS, the field `artifacts_dir_url` in
`def_types.Model` needs to point to the artifact URL, so tools know where to
download the assoicated artifacts. The URL format should be:

```sh
https://storage.googleapis.com/iree-model-artifacts/${FULL_QUALIFIED_FRAMEWORK}/${ARTIFACTS_VERSION}/${MODEL_NAME}

# For example:
# https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230723.908_1690223270/EXAMPLE_FP32_PT_BATCH1
```

We also provide templating helpers to populate `artifacts_dir_url`. See
`ARTIFACTS_DIR_URL_TEMPLATE` and how it is used in
[model_definitions.py](/common_benchmark_suite/openxla/benchmark/comparative_suite/pt/model_definitions.py).

In addition, the field `exported_model_types` also needs to be populated for
compiler-level benchmark tools (e.g. XLA-HLO) to know the supported model
formats. The list depends on what formats the model is exported to and currently
should be hard-coded with:

| Model's base framework | `exported_model_types` |
|------------------------|------------------------|
| JAX                    | `[def_types.ModelArtifactType.STABLEHLO_MLIR, def_types.ModelArtifactType.XLA_HLO_DUMP]` |
| PyTorch                | `[def_types.ModelArtifactType.LINALG_MLIR]` |
| TensorFlow             | `[def_types.ModelArtifactType.STABLEHLO_MLIR, def_types.ModelArtifactType.XLA_HLO_DUMP, def_types.ModelArtifactType.TF_SAVEDMODEL_V2]` |

You can see the example commit
[41f2e9b](https://github.com/openxla/openxla-benchmark/commit/41f2e9bb1c3598daba58ca5e7015372f4065114c)
for how to populate these fields.

After that, optionally add the new benchmarks to the benchmark list in one of
`benchmark_all.sh` under [comparative_benchmark](/comparative_benchmark), so
they will be run in the
[comparative benchmark workflow](/.github/workflows/run_comparative_benchmark.yml)
nightly.

The [xla_hlo](/comparative_benchmark/xla_hlo) is a special one as it runs
compiler-level benchmarks and uses `XLA-HLO` dump as inputs. If the model
exports `XLA_HLO_DUMP` artifact (e.g. JAX and TensorFlow models), you can also
add it to
[xla_hlo/benchmark_all.sh](/comparative_benchmark/xla_hlo/benchmark_all.sh).

Finally, submit all changes as a PR for review.
