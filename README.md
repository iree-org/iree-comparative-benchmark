# OpenXLA Benchmark

This is a home for the common benchmarking infrastructure described in the
[accompanying RFC](https://github.com/openxla/community/blob/main/rfcs/20230510-common-benchmark-suite.md).
It aims to be a common benchmark suite that is **compiler-agnostic** and can be
used in standalone comparative benchmark workflows and regression benchmarking
resident in each compiler project.

There are two components in this repository:

- [common_benchmark_suite](/common_benchmark_suite): The compiler-agnostic
  benchmark suite.
- [comparative_benchmark](/comparative_benchmark): Tools to run comparative
  benchmarks with the benchmark suite.

The *common_benchmark_suite* is standalone and should not have dependency on the
*comparative_benchmark*.

## Supported Runtimes

### Framework Level

These benchmarks are run from the Deep Learning Framework. This is the
end-to-end latency seen by the user when running the workload from a
framework such as PyTorch. Supported runtimes:

* JAX with IREE PJRT.
* JAX and Tensorflow with XLA.
* PyTorch with Inductor.

### Compiler/Library Level

These benchmarks do not include the Deep Learning Framework. This is more
reflective of the final deployment environment or AOT deployment. Supported
runtimes:

* JAX, Tensorflow, PyTorch and TFLite with IREE using MLIR input.
* JAX, Tensorflow with XLA using HLO input.
* TFLite.
* GGML (experimental).

## Supported Devices

### Server

* GPU: a2-highgpu-1g.
* CPU: c2-standard-60.
* (Retired) c2-standard-16.

### Mobile

* Pixel 6 Pro, Pixel 8 Pro.
* Motorola Edge+ (2023), Motorola Edge x30.
* (Retired) Pixel 4.

## Generated Artifacts

Most workloads are sourced from HuggingFace Transformers and are available in
PyTorch, JAX and Tensorflow. Artifacts are generated from each workload and
used as input to benchmarks. This decouples the compiler/runtime from the
framework and enables comparisons across a wider range of runtimes e.g. It is
possible to run compiler-level comparisons between IREE, XLA and TFLite using
artifacts derived from the same JAX workload.

Below is a list of artifacts that are generated from each framework:

JAX:

* StableHLO MLIR.
* XLA HLO Dump.
* Tensorflow SavedModel (through JAX2TF).
* TFLite Flatbuffer. Using Post-Training Quantization, also generates FP16,
dynamic-range quantization and INT8 variants.

PyTorch:

* Linalg MLIR (through torch-mlir).

Tensorflow:

* StableHLO MLIR.
* XLA HLO Dump.
* Tensorflow SavedModel.
* TFLite Flatbuffer. Using Post-Training Quantization, also generates FP16,
dynamic-range quantization and INT8 variants.

TFLite:

* TOSA MLIR.
* TFLite flatbuffer.

### Input/Output Data

Input and output data is also generated and saved as numpy arrays. This data can
be used downstream to test accuracy.

## Supported Workloads

Below is a list of workloads currently being benchmarked. To add more workloads,
please read "User's Guide".

### Single Model

| Framework  | Model                               | Data Type                             | Batch Sizes                            | Input Size                               |
|------------|-------------------------------------|---------------------------------------|----------------------------------------|------------------------------------------|
| JAX        | T5-Large                            | FP32, FP16, BF16                      | 1, 16, 24, 32, 48, 64, 512             | Sequence length 512                      |
| JAX        | T5-Large for Conditional-Generation | FP32                                  | 1, 16, 24, 32, 48                      | Sequence length 512                      |
| JAX        | T5-Small                            | FP32                                  | 1                                      | Sequence length 128                      |
| JAX        | Bert-Large                          | FP32, FP16, BF16                      | 1, 16, 24, 32, 48, 64, 512, 1024, 1280 | Sequence length 384                      |
| JAX        | Bert-Base                           | FP32, FP16, BF16                      | 1                                      | Input sequences 8, 32, 64, 128, 256, 512 |
| JAX        | ResNet50                            | FP32, FP16, BF16                      | 1, 8, 64, 128, 256, 2048               | Input image 3x224x224                    |
| JAX        | GPT-2 with LMHead                   | FP32                                  | 1                                      | Sequence length 512                      |
| JAX        | ViT                                 | FP32                                  | 1                                      | Input image 3x224x224                    |
| PyTorch    | Bert-Large                          | FP32, FP16                            | 1, 16, 24, 32, 48, 64, 512, 1024, 1280 | Sequence length 384                      |
| PyTorch    | ResNet50                            | FP32, FP16                            | 1, 8, 64, 128, 256, 2048               | Input image 3x224x224                    |
| Tensorflow | T5-Large                            | FP32                                  | 1, 16, 24, 32, 48, 64, 512             | Input sequence 512                       |
| Tensorflow | Bert-Large                          | FP32                                  | 1, 16, 24, 32, 48, 64, 512, 1024, 1280 | Input sequence 384                       |
| Tensorflow | RestNet50                           | FP32                                  | 1, 8, 64, 128, 256, 2048               | Input image 224x224x3                    |
| Tensorflow | EfficientNet-B7                     | FP32                                  | 1, 64, 128                             | Input image 600x600x3                    |
| TFLite     | Bert-Base                           | FP32, FP16, Dynamic-range quant, INT8 | 1                                      | Input sequences 8, 32, 64, 128, 256, 512 |
| TFLite     | ViT                                 | FP32, FP16, Dynamic-range quant, INT8 | 1                                      | Input image 3x224x224                    |

### Pipeline

Pipelines may include more than one model or control flow.

| Framework  | Pipeline          | Data Type        | Variations                                   |
|------------|-------------------|------------------|----------------------------------------------|
| JAX        | T5-Small          | FP32, FP16, BF16 | Token generation sizes: 16, 32, 64, 128, 256 |
| JAX        | Stable Diffusion  | FP32, FP16, BF16 | Input sequence 64 tokens                     |
| JAX        | GPT-2 with LMHead | FP32             | Generates 200 tokens                         |
| Tensorflow | GPT-2 with LMHead | FP32             | Generates 200 tokens                         |
| GGML       | GPT-2 with LMHead | FP32, FP16       | Generates 200 tokens                         |

## Dashboards

* IREE vs TFLite:
[Mobile](https://storage.googleapis.com/tflite-benchmark-artifacts/mobile/latest/mobile_summary.html),
[Server](https://storage.googleapis.com/tflite-benchmark-artifacts/server/latest/server_summary.html).

## User's Guide

To add new models and benchmarks, see
[Onboarding New Models and Benchmarks](/common_benchmark_suite#onboarding-new-models-and-benchmarks).

## Contacts

* [GitHub issues](https://github.com/openxla/openxla-benchmark/issues):
  Feature requests, bugs, and other work tracking
* [OpenXLA discord](https://discord.gg/UjC3bJXr7A): Daily development
  discussions with the core team and collaborators

## License

OpenXLA Benchmark is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](LICENSE) for more information.
