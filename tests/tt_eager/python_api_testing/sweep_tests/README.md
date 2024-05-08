# PyTorch Sweep Tests
This project contains infra for running sweep tests for ops in `tt_lib`.

## Description
Sweep tests are used to compare and validate `tt_lib` ops against a PyTorch golden. It mainly consists of two parts:
- A `run_test` function that runs an op through `tt_lib` and `pytorch` and compares the results.
- A lightweight wrapper around `run_test` that can perform sweeps of tests (mostly across a range of input shapes) and dump the results to output CSV's.

## Setup
Follow the `Getting Started` page to setup the `python_env`. Then, run:
```
source build/python_env/bin/activate
```

## Running tests
The tests are run through a python script:
```
python python_api_testing/sweep_tests/run_pytorch_test.py -i python_api_testing/sweep_tests/test_configs/<test_config.yaml> -o <output_folder>
```

The inputs are:
- `-i, --input-test-config`: Input test config containing list of tests to run (see below for description on what this config should look like).
- `-o, --output-folder-path`: Optional folder name to dump result CSV's in. If not specified, it will default to `pytorch_test_folder`. If this folder (or any folder specified) already exists, the tests will not run.


## Adding test configs
Here is an example of a test config:
```
---
test-list:
  eltwise-add:
    shape:
      start-shape: [1, 1, 32, 32]
      end-shape: [1, 1, 64, 64]
      interval: 32
      num-shapes: 2
      num-samples: all
      args-sampling-strategy: "all"
      method: default
    datagen:
      function: gen_rand
      args:
        low: -100
        high: 100
    comparison:
      function: comp_pcc
      args:
        pcc: 0.99
    args-gen: gen_dtype_layout_device
    sanitize-args: True
    args:
        data-layout: ["TILE"]
        data-type: ["BFLOAT16"]
        buffer-type: ["DRAM", "L1", "SYSTEM_MEMORY"]
        out-buffer-type: ["DRAM"]
    output-file: eltwise_add_sweep.csv
```

- _eltwise-add_: Maps to `tt_lib` and `pytorch` ops in `python_api_testing/sweep_tests/op_map.py`
- _shape_ and _datagen_: Passed verbatim as dictionaries to `shapes_and_datagen` function in `python_api_testing/sweep_tests/common.py`. Data from _args_ dictionary is also passed to `shapes_and_datagen` to control data-layout, data-type and memory config of input arguments (also called input config).
  - `shapes_and_datagen` is a generator that yields a matching list of shapes, datagen functions, and input configs to sweep over for the test. The datagen function is mapped to functions in `python_api_testing/sweep_tests/generation_funcs.py` and used for populating the shapes with input data.
  - In general, a sweep is performed across the start and end shapes and inputs are setup according to data-layout, data-type and memory configs defined in _args_ section. Take a look at the code for how these fields are handled, but at a high-level:
    - `start-shape` and `end-shape`: Ranges for each dim; how the sweep happens depends on `method`.
      - `shape-list`: Use this field to ignore all other parameters and use hardcoded list of shapes.
    - `interval`: Defaults to 32; step used to sweep across each dim.
    - `num-shapes`: Number of input shapes (number of inputs). Eg. unary operation has `num-shapes` equal to 1, and binary operation has number of shapes equal to 2.
    - `num-samples`: Number of generated test runs with different inputs (different shapes and configs). Defaults to `all`, which means that sweep will go through all input shapes defined by `start-shape`, `end-shape` and `interval`. If number is provided there will be (roughly) `num-samples` test runs generated.
    - `args-sampling-strategy`: Strategy how arguments to operation will be configured. If set to `all` (which is default) sweep will go through all combinations for all inputs of data-layout, data-type, and memory configs defined in _args_ dictionary. If set to `random` sweep will randomly choose data-layout, data-type, and memory configs for each input to generate `num-samples` test runs.
    - `method`: Defaults to `default`; determines how the shapes are swept across
- _comparison_: Maps to `python_api_testing/sweep_tests/comparison_funcs.py`.
- _args-gen_: Maps to `python_api_testing/sweep_tests/generation_funcs.py`. You can choose what function is used to generate arguments for the test (dtype, layout etc). For example, for `glu` you might want to generate a `dim` parameter, so default `args-gen` won't be useful.
- _sanitize-args_: `True` if `args-gen` is doing arg sanitization, `False` otherwise. If not stated, the default is `True`. When ON, args sanitization won't allow some problematic combinations of args (Eg. `BFLOAT8_B` and `ROW_MAJOR` layout). But args sanitization might be an obstacle when we want to create some more flexible tests.
- _args_: Defines how arguments to operation can be configured in terms of data-layout, data_type and memory config.
  - `data-layout`: Data layout each input argument can take. Can be TILE or ROW_MAJOR.
  - `data-type`: Data type each input argument can take. Can be BFLOAT16 or BFLOAT8_B.
  - `buffer-type`: Buffer type each input argument can take. Can be DRAM, L1, or SYSTEM_MEMORY.
  - `out-buffer-type`: Buffer type output can take. Can be DRAM, L1, or SYSTEM_MEMORY.
- _output-file_: Name of the output csv dumped inside the output folder. You can write results for additional tests to the same file if you provide the same output file path.


### _datagen_ section of test configs notes

_datagen_ section can be written for each input separately as well. Eg:

```
  ...
  datagen:
    - input_1:
      function: gen_rand
      args:
        low: -100
        high: 100
    - input_2:
      function: gen_rand
      args:
        low: -10
        high: 10
  ...
```


### _args_ section of test configs notes

_args_ section can be written for each input separately as well. Eg:

```
  comparison:
      function: comp_pcc
      args:
        pcc: 0.99
  args:
    inputs:
      - input-1:
        data-layout: ["TILE"]
        data-type: ["BFLOAT16"]
        buffer-type: ["DRAM", "L1", "SYSTEM_MEMORY"]
      - input-2:
        data-layout: ["TILE"]
        data-type: ["BFLOAT16"]
        buffer-type: ["DRAM", "L1", "SYSTEM_MEMORY"]
    out-buffer-type: ["DRAM", "L1"]
  output-file: eltwise_add_sweep.csv
```
