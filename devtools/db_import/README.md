# Database Importer

This is a small tool which we use to import benchmark results into a BigQuery database.

It is meant to be called once to batch-import pre-existing results and then be triggered
whenever a new set of results is available. The code is supposed to run in a Google
Cloud Function.

The tool is written in Python. it is similar to
[`bq load`](https://cloud.google.com/bigquery/docs/bq-command-line-tool#loading_data)
in a sense that it allows loading the data from JSON files into a BigQuery table,
but in addition it allows arbitrary data rearrangements with the help of the
[Jsonnet](https://jsonnet.org/) data processing language.

## Architecture

![Block overview](docs/images/blocks.png)

The import code lives in a cloud function which gets triggered whenever a new
file gets uploaded to a predefined GCS bucket. The tool allows to make
arbitrary changes to the data structure in the file using a
[Jsonnet](https://jsonnet.org/) expression.

Jsonnet is a superset of JSON and offers a simple way to transform structured
data. (More details below) It also allows more complex tasks like reading more
than one file and joining their contents.

The transformed result set will then be streamed into Big Query using its data
[streaming API](https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery).

## Configuration

The tool is built around a central configuration file `config.yml` in the
[YAML](https://yaml.org/) markup language.

The config file defines an arbitrary number of data processing pipelines.
Each pipeline can process events from a single GCS bucket and stream results
into a single BigQuery table. All pipelines are defined under the `pipelines`
key in the config file.

A very minimal config file could look like this:

```yaml
---
pipelines:
  my_pipeline: # `my_pipeline` is the name of a pipeline. It's what the CLI tool expects.
    bucket_name: my_source_bucket # This bucket will be registered as an event source
    table_name: my_dataset.my_destination_table # We will stream the processed data to this BigQuery table.
    cloud_function_name: the_name_of_my_cloud_function # This is what will show up in the GCP console
    rules: *my_rules # Defines the processing. Details later.
```

## Rules

As mentioned before:

- Each pipeline can process events from a single bucket.
- Data transformations can be performed using [Jsonnet](https://jsonnet.org).

![Rule processing](docs/images/rule_processing.png)

A pipeline consists of a list of rules which are attempted to be applied
in given order. Each rule defines a regular expression that must match
the file path of the triggering file. If the file path doesn't match the
next rule in the list is attempted until no rules are left.

Once a matching rule has been identified its Jsonnet transformation will
be applied and the results are streamed to BigQuery. If the transformation
or the BigQuery import fails, no other rules will be considered and the
pipeline fails (for this particular import).

Here is a very basic onfig file with a basic rule:

```yaml
---
pipelines:
  my_pipeline: # `my_pipeline` is the name of a pipeline. It's what the CLI tool expects.
    bucket_name: my_source_bucket # This bucket will be registered as an event source
    table_name: my_dataset.my_destination_table # We will stream the processed data to this BigQuery table.
    cloud_function_name: the_name_of_my_cloud_function # This is what will show up in the GCP console
    rules:
    - filepath_regex: "\\.json$" # Any JSON file
      results: *my_jsonnet_transformation # An example follows in the next section
```

## Transformations

Transformations of data has to be expressed using the
[Jsonnet](https://jsonnet.org) language.  It is a superset of JSON which
allows you to write standard JSON and sprinkle in some expressions that
reference the source file.

If your data transformations are simple enough this allows for very readable
code. If they are more complex it's possible to call native Python functions
from Jsonnet which offers an easy way to extend things.

The pipeline expects the Jsonnet output to be a list of table rows. Each row
is a `struct` which maps a column name to a value. This is a basic example:

```json
[
  {
    "col0" : 42,
    "col1" : "foo"
  },
  {
    "col0" : 128,
    "col1" : "bar"
  }
]
```

It is also a valid Jsonnet expression and could be put as-it into the `results`
field of a pipeline.  Obviously the data being imported would be static in that
case:

```yaml
---
pipelines:
  my_pipeline: # `my_pipeline` is the name of a pipeline. It's what the CLI tool expects.
    bucket_name: my_source_bucket # This bucket will be registered as an event source
    table_name: my_dataset.my_destination_table # We will stream the processed data to this BigQuery table.
    cloud_function_name: the_name_of_my_cloud_function # This is what will show up in the GCP console
    rules:
    - filepath_regex: "\\.json$" # Any JSON file
      results: |
        // Unlike JSON, Jsonnet allows comments.
        // Check out the tutorial (https://jsonnet.org/learning/tutorial.html) for more details.
        [
          { // Row0
            "col0" : 42,
            "col1" : "foo"
          },
          { // Row1
            "col0" : 128,
            "col1" : "bar"
          }
        ]
```

## Filepath captures

The `filepath_regex` regular expression allows to define named capture groups
which then can be accessed from Jsonnet.  The most common use-case is to define
at least a named capture group for the entire file path to be able to access that
from the transformation.

Example:

```yaml
---
pipelines: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\\.json)$" # Any JSON file
      results: |
        local getFilepathCapture = function(name) std.parseJson(std.extVar('filepath_captures'))[name];

        [
          {
            "filepath" : getFilepathCapture('my_arbitrary_capture_name'),
          },
        ]
```

This defines a Jsonnet function which accesses the value of a named capture
group.  This function is then called in the result expression. It assumes
the table schema has a single column `filepath` which accepts a string.

## Reading a file in a transformation

The most common use case is to actually read the file that triggered a certain
rule. The previous section showed how the file path can be obtained inside of
the Jsonnet context. For reading a file it's necessary to call the native
Python function `readFile(filepath)` which returns the contents of the file.

Example:

```yaml
---
pipelines: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\\.json)$" # Any JSON file
      results: |
        local getFilepathCapture = function(name) std.parseJson(std.extVar('filepath_captures'))[name];
        local loadJson = function(filename) std.parseJson(std.native('readFile')(filename));

        local source_json = loadJson(getFilepathCapture('my_arbitrary_capture_name'));
        /* Let's assume the loaded JSON looks like this:
        {
          'benchmark_name' : 'my_arbitrary_benchmark_name',
          'metrics' : [
            {
              'key': '...',
              'value': 42.42
            },
            // ...
          ]
        }
        */

        [
          {
            "benchmark_name" : source_json.benchmark_name,
            "metrics_key" : metric.key,
            "metrics_value" : metric.value
          },
        ] for metric in source_json.metrics
```

This example defines two boilerplate functions which allow us to read the
triggering JSON file.  Eventually we read the parsed contents into the
variable `source_json`. Afterwards we use Jsonnet's
Array comprehension to create a list of database rows.

## Snippets

The previous example shows that it is sometimes needed to define little
helper functions in Jsonnet.  To share these between rules or even pipelines
they can be put into a separate section in the config file and then imported
from the Jsonnet expression.

This is the previous example, but with the helper functions moved into the
snippets section:

```yaml
---
snippets:
  getFilepathCapture: function(name) std.parseJson(std.extVar('filepath_captures'))[name]
  loadJson: function(filename) std.parseJson(std.native('readFile')(filename))
pipelines: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\\.json)$" # Any JSON file
      results: |
        // The import key word can load functions form the snippets section
        local getFilepathCapture = import "getFilepathCapture";
        local loadJson = import "loadJson";

        local source_json = loadJson(getFilepathCapture('my_arbitrary_capture_name'));

        [
          {
            "benchmark_name" : source_json.benchmark_name,
            "metrics_key" : metric.key,
            "metrics_value" : metric.value
          },
        ] for metric in source_json.metrics
```

## Embedding

When Jsonnet expression get larger the YAML config file can become quite
confusing. That's why we support storing the Jsonnet expression in a separate
file and embed it into the config file using the `!embed` YAML tag.

Example:

```yaml
# config.yml
---
pipelines: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\\.json)$" # Any JSON file
      results: !embed my_pipeline_standard_rule.jsonnet
```

```json
// my_pipeline_standard_rule.jsonnet
[
  // Imagine an arbitrary Jsonnet expression here
]
```

## SQL statements

For deployment of the cloud function and for automatic batch import the
tool needs to be able to interact with the destination table in BigQuery.
Noteably there are 3 SQL statements that need to be provided:

1. `sql_create_table` will be executed when the destination table should
   be created
2. `sql_data_present` must be a non-mutating `SELECT` query and should
   return some data whenever there is already some data in the destination
   table, that originated from the current pipeline.
3. `sql_delete` will be executed when all the data originating from the
   current pipeline should be deleted.

Assuming the table is not shared between multiple pipelines, these 3 SQL
statements are straight forward:

```yaml
---
pipelines:
  my_pipeline:
    sql_create_table: CREATE TABLE `{dataset}.{table}` (col0 INT64, col1 STRING)
    sql_data_present: SELECT 1 FROM `{dataset}.{table}` LIMIT 1
    sql_delete: DELETE FROM `{dataset}.{table}` WHERE 1=1
```

If the destination table is shared between multiple pipelines then
both `sql_data_present` and `sql_delete` need to be more selective.
In the following example we assume that we have a column indicating
the source pipeline:

```yaml
---
pipelines:
  my_pipeline:
    sql_create_table: CREATE TABLE `{dataset}.{table}` (col0 INT64, col1 STRING, pipeline STRING)
    sql_data_present: SELECT 1 FROM `{dataset}.{table}` WHERE pipeline = "my_pipeline" LIMIT 1
    sql_delete: DELETE FROM `{dataset}.{table}` WHERE pipeline = "my_pipeline"
```

Note that this will only work if the Jsonnet rules fill the `pipeline`
field accordingly.

## The CLI tool

`cli.py` is a small CLI tool which can help with testing, initial (batch)
import, and deployment of your pipeline.

### Deployment

```bash
python cli.py deploy my_pipeline
```

The `deploy` subcommand creates or updates a cloud function for the
`my_pipeline` pipeline.  It calls the `gcloud`, `bq`, and `gsutils` CLI
tools to do its work.  So make sure, they are available and that you're
logged into the correct cloud project.

The command also creates the BigQuery table if it doesn't exist already.
But if it does, there are two options which determine how to deal with
pre-existing data:

`--force-data-deletion` deletes all pre-existing data from the table. The
pipeline config needs to supply a delete SQL statement in the `sql_delete`
field (see previous section).

`--force-data-import` will go through all the files that exist in the source
GCS bucket and triggers an import for them - even when there is already data
originating from the pipeline in the table.  This is checked by evaluating
the SQL expression in `sql_data_present`. If it returns a non-empty result
the deployment tool assumes a batch import has already happened previously
and won't attempt one - unless `--force-data-import` is given.

### Process a single file

```bash
python cli.py process -c my_pipeline --trigger this/is/the/path.json --dry-run
```

This commanded processes a given file locally and prints the resulting
rows instead of importing them into the given table.

### Verify

The verify subcommand allows testing of a pipeline without actually
deploying a cloud function or re-uploading a file to the GCS bucket.

There are 2 main uses case:

1. Test a change to the pipeline configuration
2. Test a change to the input file format

Each pipeline config allows the definition of a series of tests
(under the 'tests' node in YAML). Each test

- must have an `id` (no whitespaces please),
- can have a more descriptive `name`, (including whitespaces)
- can have a series of setup SQL statements under `setup`,
- must have one or multiple SQL-based asserts under `checks`.

The verify command will then simulate the operation of the cloud function
and will verify the integrity of the imported data by making SQL queries
to BigQuery (using the bigquery-emulator).

It also verifies whether the `sql_create_table`, `sql_data_present`, and `sql_delete`
SQL statements work by evaluating them in the process.

Example:

```yaml
pipelines:
  example_pipeline:
    bucket_name: my_bucket
    cloud_function_name: my_cloud_function
    table_name: my_dataset.my_table
    rules:
    - # Missing in this example
    tests:
    - id: my_test
      name: My Test
      setup:
      # `setup` can contain arbitrary SQL commands. You can use that to create views or user defined functions, for example.
      - CREATE VIEW my_view AS (SELECT * FROM `{dataset}.{table}`)
      triggers:
      # Note that we will process all triggers in the given order before we evaluate the `checks` below.
      # Hence the `checks` will always "see" the data from all triggers.
      - path/to/file/in/bucket.json
      - path/to/other/file/in/bucket.json
      checks:
      # The checks are arbitrary SQL commands, but it's good practice to make them non-mutating (SELECT queries only!)
      # Checks are executed in given order and the actual result is NOT processed in any way. An empty result is not an error.
      # You can use the `ERROR` SQL expression to make your query fail in all the cases you wanted - as demonstrated below:
      - SELECT CASE WHEN COUNT(*) = 42 THEN true ELSE ERROR(FORMAT('Expected 42 rows in view, but found %t'), COUNT(*))) END FROM {dataset}.my_view
      - SELECT ERROR('Field `my_field` was NULL, but not allowed') FROM {dataset}.my_view WHERE my_field IS NULL
```

The `triggers` can be overridden with the `--overwrite_triggers` command line option.
This is useful for testing whether a certain new JSON file succeeds in all the tests.

The `--benchmark_id_re` command line option also allows to filter by benchmark ID
and only execute a subset of tests. This is particular useful in conjunction
with `--overwrite_triggers` in a CI setup where we want to check the imported data
for certain invariants (things like fields must not be NULL, etc.), but not check
for an exact number of rows or even specific field values.
