# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import datetime
import json
import dataclasses
import pprint
import shutil
from types import ModuleType


from loguru import logger
import pytest

import ttnn
import ttnn.database


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, ModuleType):
        val = val.__name__
    return f"{argname}={val}"


@pytest.fixture(autouse=True)
def pre_and_post(request):
    original_config = copy.copy(ttnn.CONFIG)
    if ttnn.CONFIG_PATH is not None:
        ttnn.load_config_from_json_file(ttnn.CONFIG_PATH)
    if ttnn.CONFIG_OVERRIDES is not None:
        ttnn.load_config_from_dictionary(json.loads(ttnn.CONFIG_OVERRIDES))

    report_name = f"{request.node.nodeid}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC)"
    with ttnn.manage_config("report_name", ttnn.CONFIG.report_name or report_name):
        if ttnn.CONFIG.enable_logging and ttnn.CONFIG.report_name is not None:
            logger.debug(f"ttnn.CONFIG:\n{pprint.pformat(dataclasses.asdict(ttnn.CONFIG))}")
            report_path = ttnn.CONFIG.report_path
            if report_path.exists():
                logger.warning(f"Removing existing log directory: {report_path}")
                shutil.rmtree(report_path)
        yield

    if ttnn.database.SQLITE_CONNECTION is not None:
        ttnn.database.SQLITE_CONNECTION.close()
        ttnn.database.SQLITE_CONNECTION = None

    ttnn.tracer.disable_tracing()
    ttnn.CONFIG = original_config


def pytest_addoption(parser):
    parser.addoption("--option", action="store", default="")
    parser.addoption(
        "--no-skips", action="store_true", default=False, help="ignore pytest.skip() calls, and continue on with test"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_preparse(config, args):
    if "--no-skips" not in args:
        return

    def no_skip(*args, **kwargs):
        return

    pytest.skip = no_skip
