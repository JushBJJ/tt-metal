# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_metal.tools.profiler.merge_meta_class import MergeMetaclass
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG


class default_setup(metaclass=MergeMetaclass):
    timerAnalysis = {
        "FW_START->FW_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "ANY", "timerID": 1},
            "end": {"core": "ANY", "risc": "ANY", "timerID": 4},
        },
        "KERNEL_START->KERNEL_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "ANY", "timerID": 2},
            "end": {"core": "ANY", "risc": "ANY", "timerID": 3},
        },
        "BR_KERNEL_START->BR_KERNEL_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "BRISC", "timerID": 2},
            "end": {"core": "ANY", "risc": "BRISC", "timerID": 3},
        },
        "NC_KERNEL_START->NC_KERNEL_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "NCRISC", "timerID": 2},
            "end": {"core": "ANY", "risc": "NCRISC", "timerID": 3},
        },
        "T0_KERNEL_START->T0_KERNEL_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "TRISC_0", "timerID": 2},
            "end": {"core": "ANY", "risc": "TRISC_0", "timerID": 3},
        },
        "T1_KERNEL_START->T1_KERNEL_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "TRISC_1", "timerID": 2},
            "end": {"core": "ANY", "risc": "TRISC_1", "timerID": 3},
        },
        "T2_KERNEL_START->T2_KERNEL_END": {
            "across": "device",
            "type": "model_first_last",
            "start": {"core": "ANY", "risc": "TRISC_2", "timerID": 2},
            "end": {"core": "ANY", "risc": "TRISC_2", "timerID": 3},
        },
    }

    riscsData = {
        "BRISC": {"color": "light:g"},
        "NCRISC": {"color": "light:r"},
        "TRISC_0": {"color": "light:gray"},
        "TRISC_1": {"color": "light:gray"},
        "TRISC_2": {"color": "light:gray"},
        "TENSIX": {"color": "light:b"},
    }

    riscs = [
        "BRISC",
        "NCRISC",
        "TRISC_0",
        "TRISC_1",
        "TRISC_2",
        # "TENSIX",
    ]

    timerIDLabels = [(0, "Start"), (1, "Firmware Start"), (2, "Kernel start"), (3, "Kernel End"), (4, "Firmware End")]

    displayStats = ["Count", "Average", "Max", "Median", "Min", "Sum"]

    plotBaseHeight = 200
    plotPerCoreHeight = 100

    webappPort = 8050

    cycleRange = None
    # Example
    # cycleRange = (34.676e9, 60e9)

    intrestingCores = None
    # Example
    # intrestingCores = [(0, 0), (0, 9), (6, 9)]

    # ignoreMarkers = None
    # Example
    ignoreMarkers = [65535]

    outputFolder = f"output/device"
    deviceInputLog = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    deviceRearranged = "device_rearranged_timestamps.csv"
    deviceAnalysisData = "device_analysis_data.json"
    deviceChromeTracing = "device_chrome_tracing.json"
    devicePerfHTML = "timeline.html"
    deviceStatsTXT = "device_stats.txt"
    deviceTarball = "device_perf_results.tgz"


class test_matmul_multi_core_multi_dram(default_setup):
    timerAnalysis = {
        "Compute~": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 6},
            "end": {"risc": "BRISC", "timerID": 5},
        }
    }


class test_matmul_multi_core_multi_dram_in0_mcast(default_setup):
    timerAnalysis = {
        "NCRISC start sender -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 10},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC start reciever -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 7},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }


class test_matmul_multi_core_multi_dram_in1_mcast(default_setup):
    timerAnalysis = {
        "NCRISC start sender -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 20},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC start reciever -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 16},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }


class test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast(default_setup):
    timerAnalysis = {
        "NC_in0_s_in1_r -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 24},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_s_in1_s -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 29},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_r_in1_r -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 34},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_r_in1_s -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 39},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }


class test_full_buffer(default_setup):
    timerAnalysis = {
        "Marker Repeat": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "timerID": 5},
            "end": {"risc": "ANY", "timerID": 5},
        }
    }


class test_noc(default_setup):
    timerAnalysis = {
        "NoC For Loop": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 5},
            "end": {"risc": "NCRISC", "timerID": 6},
        }
    }
