# RFC: Benchmark ID Format

## Objective

Define the stable ID format for `def_types.BenchmarkCase`, which is the primary key of its benchmark results in the database.

## Background

The main purpose of having an ID in benchmark definitions is to help database identify if two benchmark results at different time points reference to the same benchmark, so the dashboard can track performance changes on a benchmark in the time series.

The difference between name and ID is that name consists of human-readable words to help people understand the meaning of a benchmark, while ID is its stable identifier. An ID is intentend to be random and meaningless so there is no need to update it when the benchmark is renamed. It is also required to be stable since the database uses it as priamry key to track becnhmark results. Any change on becnhmark ID will require backfill and migration of old data. 

For example, we might originally name a benchmark `T5_LARGE_FP32` while later realize there are 256 and 512 seqlen versions, so we want to rename them to `T5_LARGE_256SEQLEN_FP32` and `T5_LARGE_512SEQLEN_FP32` for clarity. As the ID of the original benchmark is a random string, we can simply rename the benchmark and keep the original ID without causing any confusion, because there is no meaningful link between the ID and the name. It also doesn't need to backfill the database because the IDs of existing benchmarks don't change. 

In general, benchmark ID gives us the freedom to update human-readable benchmark names without worrying about breaking the database.

## Proposal

TODO
