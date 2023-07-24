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

**This repository is still under early development**.

## User's Guide

To add new models and benchmarks, see
[Onboarding New Models and Benchmarks](/common_benchmark_suite#onboarding-new-models-and-benchmarks).

> TODO(pzread): Add instructions to run comparative benchmarks locally.

## Dashboard

> TODO(pzread): Add a link to the comparative benchmark dashboard.

## Contacts

* [GitHub issues](https://github.com/openxla/openxla-benchmark/issues):
  Feature requests, bugs, and other work tracking
* [OpenXLA discord](https://discord.gg/UjC3bJXr7A): Daily development
  discussions with the core team and collaborators

## License

OpenXLA Benchmark is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](LICENSE) for more information.
