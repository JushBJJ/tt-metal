# TTNN Sweep Tests

## Running all sweeps
```
python tests/ttnn/sweep_tests/run_sweeps.py
```

## Running a single sweep
```
python tests/ttnn/sweep_tests/run_sweeps.py --include add,matmul
```

## Running a single test
```
python tests/ttnn/sweep_tests/run_single_test.py --test-name add --index 0
```

## Printing report of all sweeps
```
python tests/ttnn/sweep_tests/print_report.py [--detailed]
```

## Debugging sweeps
```
python tests/ttnn/sweep_tests/run_failed_and_crashed_tests.py [--exclude add,linear] [--stepwise]
```

## Using Pytest to run sweeps all the sweeps for one operation file
```
pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>
Example for matmul: pytest tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul
```

## Using Pytest to run a single sweep test by the index
```
pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>[<operation>.py-<index-of-test-instance>]
Example for matmul: pytest tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul[matmul.py-0]
```

## Adding a new sweep test
In `tests/ttnn/sweep_tests/sweeps` add a new file `<new_file>.py`.

The file must contain:
- `parameters` dictionary from a variable to the list of values to sweep
- `skip` function for filtering out unwanted combinations. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the reason to skip the test.
- `is_expected_to_fail` function for marking the test as expected to fail. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the expected exception.
- `run` function for running the test. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the error message.

For example, let's add `tests/ttnn/sweep_tests/sweeps/to_and_from_device.py`:
```python

import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

parameters = {
    "height": [1, 32],
    "width": [1, 32],
}

def skip(height, width) -> Tuple[bool, Optional[str]]:
    return False, None

def is_expected_to_fail(height, width) -> Tuple[bool, Optional[str]]:
    return False

def run(height, width, *, device) -> Tuple[bool, Optional[str]]:
    torch_tensor = torch.zeros((height, width))

    tensor = ttnn.from_torch(torch_tensor, device=device)
    tensor = ttnn.to_torch(tensor)

    return check_with_pcc(torch_tensor, tensor)

```
