# Database Importer

This is a small tool which we use to import benchmark results into a BigQuery database.

It is meant to be called once to batch-import pre-existing results and then be triggered
whenever a new set of results is available. The code is supposed to run in a Google Cloud Function.

The tool is written in Python.

## Architecture

![Block overview](docs/images/blocks.png)

The import code lives in a cloud function which gets triggered whenever a new file gets uploaded to
a predefined GCS bucket. The tool allows to make arbitrary changes to the data structure in the file
using a [Jsonnet](https://jsonnet.org/) expression.

Jsonnet is a superset of JSON and offers a simple way to transform structured data. (More details below)
It also allows more complex tasks like reading more than one file and joining their contents.

The transformed result set will then be streamed into Big Query.

## Configuration

The tool is built around a central configuration file `config.yml` in the [YAML](https://yaml.org/) markup language.

The config file defines an arbitrary number of data processing pipelines. Each pipeline can process events from a single
GCS bucket and stream results into a single BigQuery table. All pipelines are defined under the `cloud_functions` key in the config file.

A very minimal config file could look like this:

```yaml
---
cloud_functions:
  my_pipeline: # `my_pipeline` is the name of a pipeline. It's what the CLI tool expects.
    bucket_name: my_source_bucket # This bucket will be registered as an event source
    table_name: my_dataset.my_destination_table # We will stream the processed data to this BigQuery table.
    cloud_function_name: the_name_of_my_cloud_function # This is what will show up in the GCP console
    rules: *my_rules # Defines the processing. Details later.
```

## Rules

As mentioned before:
- Each cloud function (or pipeline) can process events from a single bucket.
- Data transformations can be performed using [Jsonnet](https://jsonnet.org).

![Rule processing](docs/images/rule_processing.png)

A pipeline consists of a list of rules which are attempted to be applied in given
order. Each rule defines a regular expression that must match the file path of
a given file. If the file path doesn't match the next rule in the list is attempted
until no rules are left.

Once a matching rule has been identified its Jsonnet transformation will be applied and
the results streamed to BigQuery. If the transformation or the BigQuery import fails,
no other rules will be considered.

Here is an config file example with a basic rule:

```yaml
---
cloud_functions:
  my_pipeline: # `my_pipeline` is the name of a pipeline. It's what the CLI tool expects.
    bucket_name: my_source_bucket # This bucket will be registered as an event source
    table_name: my_dataset.my_destination_table # We will stream the processed data to this BigQuery table.
    cloud_function_name: the_name_of_my_cloud_function # This is what will show up in the GCP console
    rules:
    - filepath_regex: "\.json$" # Any JSON file
      results: *my_jsonnet_transformation # An example follows in the next section
```


## Transformations

Transformations of data has to be expressed using the [Jsonnet](https://jsonnet.org) language.
It is a superset of JSON which allows you to write standard JSON and sprinkle in some
expressions that reference the source file.

If your data transformations are simple enough this allows for very readable code. If they
are more complex it's possible to call native Python functions from Jsonnet which offers
an easy way to extend things.

The pipeline expects the Jsonnet output to be a list of table rows. Each row is a `struct`
which maps a column name to a value. This is a basic example:

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

It is also a valid Jsonnet expression and could be put as-it into the `results` field of a pipeline:


```yaml
---
cloud_functions:
  my_pipeline: # `my_pipeline` is the name of a pipeline. It's what the CLI tool expects.
    bucket_name: my_source_bucket # This bucket will be registered as an event source
    table_name: my_dataset.my_destination_table # We will stream the processed data to this BigQuery table.
    cloud_function_name: the_name_of_my_cloud_function # This is what will show up in the GCP console
    rules:
    - filepath_regex: "\.json$" # Any JSON file
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

The `filepath_regex` regular expression allows to define named capture groups which then can be accessed from Jsonnet.
The most common use-case is to define at least a named capture group for the entire file path to be able to access that
from the transformation.

Example:
```yaml
---
cloud_functions: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\.json)$" # Any JSON file
      results: |
        local getFilepathCapture = function(name) std.parseJson(std.extVar('filepath_captures'))[name];

        [
          {
            "filepath" : getFilepathCapture('my_arbitrary_capture_name'),
          },
        ]
```

This defines a Jsonnet function which accesses the value of a named capture group. This function is then called
in the result expression. It assumes the table schema has a single column `filepath` which accepts a string.


## Reading a file in a transformation

The most common use case is to actually read the file that triggered a certain rule. The previous section
showed how the file path can be obtained inside of the Jsonnet context. For reading a file it's necessary
to call the native Python function `readFile(filepath)` which returns the contents of the file.

Example:
```yaml
---
cloud_functions: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\.json)$" # Any JSON file
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

This example defines two boilerplate functions which allow us to read the triggering JSON file.
Eventually we read the parsed contents into the variable `source_json`. Afterwards we use Jsonnet's
Array comprehension to create a list of database rows.

## Snippets

The previous example shows that it is sometimes needed to define little helper functions in Jsonnet. 
To share these between rules or even pipelines they can be put into a separate section in the config
file and then imported from the Jsonnet expression.

This is the previous example, but with the helper functions moved into the snippets section:

```yaml
---
snippets:
  getFilepathCapture: function(name) std.parseJson(std.extVar('filepath_captures'))[name]
  loadJson: function(filename) std.parseJson(std.native('readFile')(filename))
cloud_functions: # This config is not complete - some keys are missing
  my_pipeline:
    rules:
    - filepath_regex: "^(?P<my_arbitrary_capture_name>.*\.json)$" # Any JSON file
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

## The CLI tool

`cli.py` is a small CLI tool which can help with testing, initial (batch) import, and deployment of your pipeline.


### Deployment

```bash
python cli.py deploy my_pipeline
```

The `deploy` subcommand creates a cloud function for the `my_pipeline` pipeline.
It calls the `gcloud`, `bq`, and `gsutils` CLI tools to do its work.
So make sure, they are available and that you're logged into the correct cloud project.

### Process a single file

```bash
python cli.py process -c my_pipeline --trigger this/is/the/path.json --dry-run
```

This commanded processes a given file locally and prints the resulting rows instead of importing them into the given table.

