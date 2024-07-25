#!/bin/bash

echo "[Info] Showing root"
ls -hal
echo "[Info] Showing ttnn"
ls -hal ttnn
echo "[Info] Showing site-packages"
ls -hal python_env/lib/python3.8/site-packages
find python_env/lib/python3.8/site-packages -name *.egg* | xargs -n 1 -I {} bash -c 'echo {} && cat {}'
which pip
which pip3
which python3
python3 -c 'import scripts; import ttnn; import loguru; print(loguru.__file__); print(scripts.__file__); print(ttnn.__file__); print(ttnn._ttnn.__file__);'
