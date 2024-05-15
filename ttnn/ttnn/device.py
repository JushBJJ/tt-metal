# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

import tt_lib as ttl
import ttnn


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


# TODO: Device = ttnn._ttnn.Device
Device = ttl.device.Device
Device.core_grid = property(get_device_core_grid)


def open_device(device_id: int, l1_small_size: int = ttl.device.DEFAULT_L1_SMALL_SIZE):
    """
    open_device(device_id: int) -> ttnn.Device:

    Open a device with the given device_id. If the device is already open, return the existing device.
    """
    return ttnn._ttnn.device.open_device(device_id=device_id, l1_small_size=l1_small_size)


def close_device(device):
    """
    close_device(device: ttnn.Device) -> None:

    Close the device and remove it from the device cache.
    """
    synchronize_device(device)
    ttnn._ttnn.device.close_device(device)


def enable_program_cache(device):
    ttnn._ttnn.device.enable_program_cache(device)


def disable_and_clear_program_cache(device):
    ttnn._ttnn.device.disable_and_clear_program_cache(device)


def synchronize_device(device):
    """
    synchronize_device(device: ttnn.Device) -> None:

    Synchronize the device with host by waiting for all operations to complete.
    """
    ttl.device.Synchronize(device)


def begin_trace_capture(device, trace_buff_size, cq_id=0):
    return ttnn._ttnn.device.begin_trace_capture(device, trace_buff_size, cq_id)


def end_trace_capture(device, trace_id, cq_id=0):
    ttnn._ttnn.device.end_trace_capture(device, trace_id, cq_id)


def execute_trace(device, trace_id, cq_id=0, blocking=True):
    ttnn._ttnn.device.execute_trace(device, trace_id, cq_id, blocking)


def release_trace(device, trace_id, cq_id=0):
    ttnn._ttnn.device.release_trace(device, trace_id, cq_id)


@contextlib.contextmanager
def manage_device(device_id: int):
    """
    manage_device(device_id: int) -> ttnn.Device:

    Context manager for opening and closing a device.
    """
    device = open_device(device_id=device_id)
    try:
        yield device
    finally:
        close_device(device)


def dump_device_memory_state(device, prefix=""):
    ttl.device.DumpDeviceMemoryState(device, prefix)


__all__ = []
