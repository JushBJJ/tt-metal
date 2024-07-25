#!/bin/bash

python3 -c 'import scripts; import ttnn; import loguru; print(scripts.__file__); print(ttnn.__file__); print(ttnn._ttnn.__file__); print(loguru.__file__);'
