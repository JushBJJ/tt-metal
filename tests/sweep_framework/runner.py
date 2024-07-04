# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime
import uuid
from multiprocessing import Process, Queue
from queue import Empty
import subprocess
from ttnn import *
from serialize import *
from test_status import TestStatus
import architecture
from elasticsearch import Elasticsearch

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        raise RuntimeError("Couldn't get git hash!") from e


def run(test_module, input_queue, output_queue):
    device = ttnn.open_device(0)
    try:
        while True:
            test_vector = input_queue.get(block=True, timeout=1)
            test_vector = deserialize_vector(test_vector)
            try:
                status, message = test_module.run(**test_vector, device=device)
            except Exception as e:
                status, message = False, str(e)
            output_queue.put([status, message])
    except Empty as e:
        ttnn.close_device(device)
        exit(0)


def execute_batch(test_module, test_vectors):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    for test_vector in test_vectors:
        result = dict()
        if p is None:
            p = Process(target=run, args=(test_module, input_queue, output_queue))
            p.start()
        try:
            input_queue.put(test_vector)
            response = output_queue.get(block=True, timeout=5)
            status, message = response[0], response[1]
            result["status"] = TestStatus.PASS if status else TestStatus.FAIL_ASSERT_EXCEPTION
            result["message"] = message
        except Empty as e:
            print(f"SWEEPS: TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
            p.terminate()
            p = None
            smi_dir = architecture.tt_smi_path(ARCH)
            smi_process = subprocess.run([smi_dir, "-tr", "0"])
            if smi_process.returncode == 0:
                print("SWEEPS: TT-SMI Reset Complete Successfully")
            result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results.append(result)
    if p is not None:
        p.join()

    return results


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "batch_name", "vector_id"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for field in info_field_names:
            header[field] = vector.pop(field)
        vector.pop("timestamp")
        header_info.append(header)
    return header_info, test_vectors


def run_sweeps(module_name, batch_name):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if not module_name:
        for file in sorted(sweeps_path.glob("*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            test_module = importlib.import_module("sweeps." + sweep_name)
            vector_index = sweep_name + "_test_vectors"
            print(f"SWEEPS: Executing tests for module {sweep_name}...")
            try:
                response = client.search(
                    index=vector_index, aggregations={"batches": {"terms": {"field": "batch_name.keyword"}}}
                )
                batches = [batch["key"] for batch in response["aggregations"]["batches"]["buckets"]]
                if len(batches) == 0:
                    continue

                for batch in batches:
                    print(f"SWEEPS: Executing tests for module {sweep_name}, batch {batch}.")
                    response = client.search(index=vector_index, query={"match": {"batch_name": batch}}, size=10000)
                    test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
                    test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
                    for i in range(len(test_ids)):
                        test_vectors[i]["vector_id"] = test_ids[i]

                    header_info, test_vectors = sanitize_inputs(test_vectors)
                    results = execute_batch(test_module, test_vectors)
                    print(f"SWEEPS: Completed tests for module {sweep_name}, batch {batch}.")
                    print(f"SWEEPS: Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
            except Exception as e:
                print(e)
                continue

    else:
        test_module = importlib.import_module("sweeps." + module_name)
        vector_index = module_name + "_test_vectors"

        try:
            if not batch_name:
                try:
                    response = client.search(
                        index=vector_index, aggregations={"batches": {"terms": {"field": "batch_name.keyword"}}}
                    )
                    batches = [batch["key"] for batch in response["aggregations"]["batches"]["buckets"]]
                    if len(batches) == 0:
                        return

                    for batch in batches:
                        response = client.search(index=vector_index, query={"match": {"batch_name": batch}}, size=10000)
                        test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
                        header_info, test_vectors = sanitize_inputs(test_vectors)
                        results = execute_batch(test_module, test_vectors)
                        export_test_results(header_info, results)
                except Exception as e:
                    print(e)
                    return
            else:
                response = client.search(index=vector_index, query={"match": {"batch_name": batch}}, size=10000)
                test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]

                header_info, test_vectors = sanitize_inputs(test_vectors)
                results = execute_batch(test_module, test_vectors)
                export_test_results(header_info, results)
        except Exception as e:
            print(e)

    client.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(header_info, results):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))
    sweep_name = header_info[0]["sweep_name"]
    results_index = sweep_name + "_test_results"

    try:
        git = git_hash()
        for result in results:
            result["git_hash"] = git
    except:
        pass

    for i in range(len(results)):
        result = header_info[i]
        for elem in results[i].keys():
            result[elem] = serialize(results[i][elem])
        client.index(index=results_index, id=uuid.uuid4(), body=result)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )

    parser.add_argument(
        "--elastic", required=False, help="Elastic Connection String for the vector and results database."
    )
    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted.")
    parser.add_argument("--batch-name", required=False, help="Batch of Test Vectors to run, or all tests if omitted.")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"],
        help="Device architecture",
    )

    args = parser.parse_args(sys.argv[1:])

    if not args.module_name and args.batch_name:
        parser.print_help()
        print("ERROR: Module name is required if batch id is specified.")
        exit(1)

    global ELASTIC_CONNECTION_STRING
    ELASTIC_CONNECTION_STRING = args.elastic if args.elastic else "http://localhost:9200"

    global ARCH
    ARCH = architecture.str_to_arch(args.arch)

    run_sweeps(args.module_name, args.batch_name)
