# RFC: Stable Benchmark Identifiers

## Objective

Define the stable identifiers for benchmarks so benchmark results can be tracked
in time series in databases even if benchmark names are changed over time.

## Background

Tracking performance changes over time is an important aspect in benchmarking,
which is done by running the same benchmarks periodically and tracking their
results in a database. A benchmark suite can contain a few dozens or hundreds of
benchmarks. To form time series of each benchmark, databases group results by
the benchmark name or identifiable metadata. That is, these metadata are seen as
the unique "primary key" of benchmarks.

Although it doesn't happen frequently, the benchmark name or the values of
identifiable metadata might change over time. For example, the model name of a
benchmark might change from `T5_LARGE_FP32` to `T5_LARGE_SEQLEN256_FP32` because
of new added model variants with different sequence lengths. If the model name
is part of the benchmark name or identifiable metadata, databases will treat the
renamed benchmark as a new one. Existing data points from the original benchmark
will need to backfill with the new name.

However, instead of backfilling every time a name is changed, another solution
is having an ID in addition to the human-readable name. For example, a model can
have the name to summarize its characteristics while also having a meaningless
random UUID. This makes it possible to only update the model name while keeping
the original ID. The database can use IDs as primary key so name changes won't
affect tracking.

## Proposal

This RFC proposes to use the key-value pairs of sub-components' IDs (e.g., model
ID, input data ID) of a benchmark as the identifiers. Each distinct benchmark
will have the unique key-value pairs of identifiers.

Component's ID can be a random UUID if it is a simple definition (e.g, model,
input data) or another key-value pairs if the component also consists of
sub-components. The rule of thumb is that hand-written definitions in the
benchmark suite usually have hard-coded random UUIDs while generated definitions
(e.g., combinations of models and input data) are compositions of sub-components
and combine their IDs as identifiers.

The identifiers of the current benchmark definition `BenchmarkCase` can be
written in JSON as:

```json
{
  "model_id": "${model uuid}",
  "input_id": "${input data uuid}",
  "expected_output_id": "${expected output uuid}",
  "target_device_id": "${device spec uuid}",
}
```

The Python definitions will be:

```py
@dataclass(frozen=True)
class BenchmarkIdentifiers:
  model_id: str
  input_id: str
  expected_output_id: str
  target_device_id: str

@dataclass(frozen=True)
class BenchmarkCase:
  # Unique human-readable name.
  name: str
  # Benchmark identifiers
  identifiers: BenchmarkIdentifiers

  model: Model
  input_data: ModelTestData
  expected_output: ModelTestData
  target_device: DeviceSpec

  @classmethod
  def build(cls, model, input_data, expected_output, target_device):
    identifiers = BenchmarkIdentifiers(
      model.id,
      input_data.id,
      expected_output.id,
      target_device.id,
    )
    return cls(identifiers=identifiers, ...)
```

### Identify benchmarks in databases

The benchmark identifiers are reported in benchmark results and uploaded to the
database. It can be directly stored as a JSON field if the database supports
querying JSON, or stored in multiple columns.

### Process to add and remove identifiers

To add or remove an identifier, the database is first updated to handle the
added or deleted identifier field. For the new field on the existing data, the
database can choose a default value for it. Backfills can be avoided if all
existing benchmarks also use the same default value for the new identifier. When
removing an identifier, the database can simply update queries to ignore it in
the existing data.

Once the database can handle the changes in the identifier fields, benchmark
tools are updated to report the new identifiers.

## Alternatives Considered

The initial prototype of common benchmark suite used a string field
`benchmark_id` to identify a benchmark. The format was:

```
models/${model_id}/inputs/${input_id}/expected_outputs/${expected_output_id}/target_devices/${device_spec_id}
```

String ID plays nicely as the keys of maps (dictionaries) in codes and the
primary key in databases. The downside is that when adding or removing
identifiers from the ID, it's more complicated to update the existing data in
the database. Because that changes the string order/format of the ID, which
requires to backfill the `benchmark_id` field on existing data.
