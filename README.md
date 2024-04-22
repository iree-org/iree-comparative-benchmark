# Mobile Benchmark

This repository is focused on on-device ML benchmarks, using IREE codegen.

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

### Compiler/Library Level

These benchmarks do not include the Deep Learning Framework. This is more
reflective of the final deployment environment or AOT deployment. Supported
runtimes:

* JAX, Tensorflow, PyTorch and TFLite with IREE using MLIR input.
* TFLite.
* GGML (experimental).

## Supported Devices

### Mobile

* Pixel 6 Pro, Pixel 8 Pro.
* Motorola Edge+ (2023), Motorola Edge x30.
* (Retired) Pixel 4.

## Generated Artifacts

Most workloads are sourced from HuggingFace Transformers and are available in
PyTorch, JAX and Tensorflow. Artifacts are generated from each workload and
used as input to benchmarks.

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
| TFLite     | Bert-Base                           | FP32, FP16, Dynamic-range quant, INT8 | 1                                      | Input sequences 8, 32, 64, 128, 256, 512 |
| TFLite     | ViT                                 | FP32, FP16, Dynamic-range quant, INT8 | 1                                      | Input image 3x224x224                    |

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

## License

OpenXLA Benchmark is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](LICENSE) for more information.
