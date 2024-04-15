#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

run_additional_T3000_test(){
    remove_default_log_locations
    mkdir -p $PROFILER_ARTIFACTS_DIR

    ./tt_metal/tools/profiler/profile_this.py -c "pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_all_gather.py::test_all_gather_on_t3000_post_commit[mem_config0-input_dtype0-8-1-input_shape1-0-layout1]" > $PROFILER_ARTIFACTS_DIR/test_out.log

    if cat $PROFILER_ARTIFACTS_DIR/test_out.log | grep "SKIPPED"
    then
        echo "No verification as test was skipped"
    else
        echo "Verifying test results"
        runDate=$(ls $PROFILER_OUTPUT_DIR/)
        LINE_COUNT=9 #1 header + 8 devices
        res=$(verify_perf_line_count "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$LINE_COUNT")
        echo $res
    fi
    cat $PROFILER_ARTIFACTS_DIR/test_out.log
}

run_profiling_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    echo "Make sure this test runs in a build with ENABLE_PROFILER=1 ENABLE_TRACY=1"

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    run_additional_T3000_test

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py -vvv

    remove_default_log_locations

    $PROFILER_SCRIPTS_ROOT/profile_this.py -c "pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py::test_run_matmul_test[BFLOAT16-input_shapes0]"

    runDate=$(ls $PROFILER_OUTPUT_DIR/)

    CORE_COUNT=7
    res=$(verify_perf_column "$PROFILER_OUTPUT_DIR/$runDate/ops_perf_results_$runDate.csv" "$CORE_COUNT" "1" "1")
    echo $res

    remove_default_log_locations
}

run_post_proc_test(){
    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_logs.py -vvv
}

cd $TT_METAL_HOME

if [[ $1 == "PROFILER" ]]; then
    run_profiling_test
elif [[ $1 == "POST_PROC" ]]; then
    run_post_proc_test
else
    run_profiling_test
    run_post_proc_test
fi
