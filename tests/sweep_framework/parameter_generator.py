# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import importlib
import pathlib
import datetime
import os
import uuid

from architecture import str_to_arch
from permutations import *
from sql_utils import *
from serialize import serialize
from elasticsearch import Elasticsearch

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Generate vectors from module parameters
def generate_vectors(test_module, arch):
    parameters = test_module.parameters

    vectors = []
    for batch in parameters:
        batch_vectors = list(permutations(parameters[batch]))
        for v in batch_vectors:
            v["batch_name"] = batch
        vectors += batch_vectors
    return vectors


# Perform any post-gen validation to the resulting vectors.
def validate_vectors(vectors) -> None:
    pass


# Output the individual test vectors.
def export_test_vectors(module_name, vectors):
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))

    # TODO: Duplicate batch check?

    index_name = module_name + "_test_vectors"

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i in range(len(vectors)):
        vector = dict()
        vector["sweep_name"] = module_name
        vector["timestamp"] = current_time
        for elem in vectors[i].keys():
            vector[elem] = serialize(vectors[i][elem])
        client.index(index=index_name, id=uuid.uuid4(), body=vector)


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name, arch):
    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            module_name = str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))[:-3]
            test_module = importlib.import_module("sweeps." + module_name)
            vectors = generate_vectors(test_module, arch)
            validate_vectors(vectors)
            export_test_vectors(module_name, vectors)
    else:
        vectors = generate_vectors(importlib.import_module("sweeps." + module_name), arch)
        validate_vectors(vectors)
        export_test_vectors(module_name, vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument("--elastic", required=False, help="Elastic Connection String for vector database.")
    parser.add_argument("--seed", required=False, default=0, help="Seed for random value generation")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"],
        help="Device Architecture",
    )

    args = parser.parse_args(sys.argv[1:])

    global ELASTIC_CONNECTION_STRING
    ELASTIC_CONNECTION_STRING = args.elastic if args.elastic else "http://localhost:9200"

    generate_tests(args.module_name, str_to_arch(args.arch))
