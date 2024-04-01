# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
import sqlite3
from typing import Optional

from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import Plot, ColumnDataSource, LinearAxis, CustomJSTickFormatter, NumeralTickFormatter, Rect, Range1d
from bokeh.models.tools import WheelZoomTool, PanTool, ResetTool, ZoomInTool, ZoomOutTool, HoverTool

from flask import Flask, render_template
import numpy as np
import torch

import ttnn


@dataclasses.dataclass
class Device:
    device_id: int
    num_y_cores: int
    num_x_cores: int
    num_y_compute_cores: int
    num_x_compute_cores: int
    worker_l1_size: int
    l1_num_banks: int
    l1_bank_size: int
    address_at_first_l1_bank: int
    address_at_first_l1_cb_buffer: int
    num_banks_per_storage_core: int
    num_compute_cores: int
    num_storage_cores: int
    total_l1_memory: int
    total_l1_for_tensors: int
    total_l1_for_interleaved_buffers: int
    total_l1_for_sharded_buffers: int
    cb_limit: int


@dataclasses.dataclass
class Operation:
    operation_id: int
    name: str
    duration: float
    matches_golden: Optional[bool]
    desired_pcc: Optional[float]
    actual_pcc: Optional[float]


@dataclasses.dataclass
class Buffer:
    operation_id: int
    device_id: int
    address: int
    max_size_per_bank: int
    buffer_type: int


@dataclasses.dataclass
class BufferPage:
    operation_id: int
    device_id: int
    address: int
    core_y: int
    core_x: int
    bank_id: int
    page_index: int
    page_address: int
    page_size: int
    buffer_type: int


def query_device_by_id(device_id):
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
    for row in cursor.fetchall():
        operation = Device(*row)
        return operation

    sqlite_connection.close()


def query_operation_by_id(operation_id):
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        operation = Operation(*row)
        return operation

    sqlite_connection.close()


def query_operations():
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations")
    for row in cursor.fetchall():
        operation = Operation(*row)
        yield operation

    sqlite_connection.close()


def query_buffers(operation_id):
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM buffers WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield Buffer(*row)

    sqlite_connection.close()


def query_buffer_pages(operation_id):
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM buffer_pages WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield BufferPage(*row)

    sqlite_connection.close()


ADDRESS_TO_ID = {}
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black"]


def duration_to_string(duration):
    if duration < 1e-6:
        return f"{duration * 1e9:.2f} ns"
    elif duration < 1e-3:
        return f"{duration * 1e6:.2f} us"
    elif duration < 1:
        return f"{duration * 1e3:.2f} ms"
    return f"{duration:.2f} s"


def duration_to_color(duration):
    if duration < 1e-6:
        return "green"
    elif duration < 1e-3:
        return "yellow"
    elif duration < 1:
        return "orange"
    return "red"


app = Flask(__name__)


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/operations")
def operations():
    operations = list(query_operations())

    return render_template(
        "operations.html",
        operations=operations,
        duration_to_string=duration_to_string,
        duration_to_color=duration_to_color,
    )


@app.route("/operations_with_l1_buffer_usage")
def operations_with_l1_buffer_usage():
    operations = list(query_operations())

    l1_reports = {}
    for operation in operations:
        l1_reports[operation.operation_id] = create_summarized_l1_buffer_plot(operation.operation_id)

    return render_template(
        "operations_with_l1_buffer_usage.html",
        operations=operations,
        duration_to_string=duration_to_string,
        duration_to_color=duration_to_color,
        l1_reports=l1_reports,
    )


def create_summarized_l1_buffer_plot(operation_id):
    glyph_y_location = 0
    glyph_height = 1

    buffers = list(query_buffers(operation_id))
    if len(buffers) == 0:
        return "", "There are no L1 Buffers!"
    device_ids = set(buffer.device_id for buffer in buffers)
    if len(device_ids) != 1:
        return "", "Cannot visualize buffer plot for multiple devices!"
    device_id = device_ids.pop()
    device = query_device_by_id(device_id)

    l1_size = device.worker_l1_size

    memory_glyph_y_location = [glyph_y_location]
    memory_glyph_x_location = [l1_size // 2]
    memory_height = [glyph_height]
    memory_width = [l1_size]
    memory_color = ["white"]
    memory_line_color = ["black"]

    memory_data_source = ColumnDataSource(
        dict(
            glyph_y_location=memory_glyph_y_location,
            glyph_x_location=memory_glyph_x_location,
            glyph_height=memory_height,
            glyph_width=memory_width,
            color=memory_color,
            line_color=memory_line_color,
        )
    )

    buffers_glyph_y_location = []
    buffers_glyph_x_location = []
    buffers_height = []
    buffers_width = []
    buffers_color = []
    buffers_max_size_per_bank = []
    buffers_address = []

    for buffer in query_buffers(operation_id):
        if buffer.address not in ADDRESS_TO_ID:
            ADDRESS_TO_ID[buffer.address] = len(ADDRESS_TO_ID)

        buffers_address.append(buffer.address)
        buffers_max_size_per_bank.append(buffer.max_size_per_bank)
        buffers_glyph_y_location.append(glyph_y_location)
        buffers_glyph_x_location.append(buffer.address + buffer.max_size_per_bank // 2)
        buffers_height.append(glyph_height)
        buffers_width.append(buffer.max_size_per_bank)
        buffers_color.append(COLORS[ADDRESS_TO_ID[buffer.address] % len(COLORS)])

    buffers_glyph_x_location = np.asarray(buffers_glyph_x_location)
    buffers_glyph_y_location = np.asarray(buffers_glyph_y_location)
    buffers_height = np.asarray(buffers_height)
    buffers_width = np.asarray(buffers_width)
    buffers_data_source = ColumnDataSource(
        dict(
            glyph_y_location=buffers_glyph_y_location,
            glyph_x_location=buffers_glyph_x_location,
            glyph_height=buffers_height,
            glyph_width=buffers_width,
            color=buffers_color,
            address=buffers_address,
            max_size_per_bank=buffers_max_size_per_bank,
        )
    )

    plot = Plot(title=None, width=1000, height=100, min_border=0, toolbar_location="below")

    xaxis = LinearAxis()
    plot.x_range = Range1d(0, l1_size)
    plot.add_layout(xaxis, "below")
    plot.xaxis.axis_label = "L1 Address Space"
    plot.xaxis.formatter = NumeralTickFormatter(format="0000000")

    memory_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color="line_color",
        fill_color="color",
    )
    plot.add_glyph(memory_data_source, memory_glyph)

    buffer_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color="black",
        fill_color="color",
    )
    buffer_renderer = plot.add_glyph(buffers_data_source, buffer_glyph)

    plot.add_tools(
        WheelZoomTool(),
        PanTool(),
        ResetTool(),
        ZoomInTool(),
        ZoomOutTool(),
        HoverTool(
            renderers=[buffer_renderer],
            tooltips=[("Address", "@address"), ("Max Size Per Bank", "@max_size_per_bank")],
        ),
    )
    return components(plot)


def create_detailed_l1_buffer_plot(operation_id):
    buffers = list(query_buffers(operation_id))
    device_ids = set(buffer.device_id for buffer in buffers)
    if len(buffers) == 0:
        return "", "There are no L1 Buffers!"
    if len(device_ids) != 1:
        return "", "Cannot visualize buffer plot for multiple devices!"
    device_id = device_ids.pop()
    device = query_device_by_id(device_id)
    l1_size = device.worker_l1_size

    core_grid = [device.num_y_cores, device.num_x_cores]
    num_cores = math.prod(core_grid)
    core_glyph_height = 100
    core_glyph_width = 10
    core_glyph_y_offset = core_glyph_height + 20
    core_glyph_x_offset = core_glyph_width + 10

    cores_y = []
    cores_x = []
    cores_glyph_y_location = []
    cores_glyph_x_location = []
    for core_y in range(core_grid[0]):
        for core_x in range(core_grid[1]):
            cores_y.append(core_y)
            cores_x.append(core_x)
            cores_glyph_y_location.append(core_y * core_glyph_y_offset)
            cores_glyph_x_location.append(core_x * core_glyph_x_offset)

    cores_glyph_y_location = np.asarray(cores_glyph_y_location)
    cores_glyph_x_location = np.asarray(cores_glyph_x_location)
    cores_height = np.full((num_cores,), core_glyph_height)
    cores_width = np.full((num_cores,), core_glyph_width)
    cores_data_source = ColumnDataSource(
        dict(
            glyph_y_location=cores_glyph_y_location,
            glyph_x_location=cores_glyph_x_location,
            glyph_height=cores_height,
            glyph_width=cores_width,
            core_y=cores_y,
            core_x=cores_x,
        )
    )

    buffer_pages_glyph_y_location = []
    buffer_pages_glyph_x_location = []
    buffer_pages_height = []
    buffer_pages_width = []
    buffer_pages_color = []

    for buffer_page in query_buffer_pages(operation_id):
        if buffer_page.address not in ADDRESS_TO_ID:
            ADDRESS_TO_ID[buffer_page.address] = len(ADDRESS_TO_ID)

        buffer_page_glyph_y_location = buffer_page.core_y * core_glyph_y_offset + core_glyph_height // 2
        buffer_page_glyph_y_location = (
            buffer_page_glyph_y_location - (l1_size - buffer_page.page_address) / (l1_size) * core_glyph_height
        )
        buffer_page_glyph_x_location = buffer_page.core_x * core_glyph_x_offset

        buffer_page_glyph_height = buffer_page.page_size / l1_size * core_glyph_height

        buffer_pages_glyph_y_location.append(buffer_page_glyph_y_location)
        buffer_pages_glyph_x_location.append(buffer_page_glyph_x_location)
        buffer_pages_height.append(buffer_page_glyph_height)
        buffer_pages_width.append(core_glyph_width)
        buffer_pages_color.append(COLORS[ADDRESS_TO_ID[buffer_page.address] % len(COLORS)])

    buffer_pages_glyph_x_location = np.asarray(buffer_pages_glyph_x_location)
    buffer_pages_glyph_y_location = np.asarray(buffer_pages_glyph_y_location)
    buffer_pages_height = np.asarray(buffer_pages_height)
    buffer_pages_width = np.asarray(buffer_pages_width)
    buffer_pages_data_source = ColumnDataSource(
        dict(
            glyph_y_location=buffer_pages_glyph_y_location,
            glyph_x_location=buffer_pages_glyph_x_location,
            glyph_height=buffer_pages_height,
            glyph_width=buffer_pages_width,
            color=buffer_pages_color,
        )
    )

    plot = Plot(title=None, width=1000, height=1000, min_border=0, toolbar_location="below")

    plot.y_range = Range1d(-100, 1200)
    plot.x_range = Range1d(-10, 250)

    xaxis = LinearAxis()
    plot.add_layout(xaxis, "below")

    yaxis = LinearAxis()
    plot.add_layout(yaxis, "left")

    plot.yaxis.axis_label = "Y Axis"
    plot.xaxis.axis_label = "X Axis"

    plot.yaxis.ticker.desired_num_ticks = 1
    plot.yaxis.formatter = CustomJSTickFormatter(
        code=f"""
        return "";
    """
    )
    plot.xaxis.ticker.desired_num_ticks = 1
    plot.xaxis.formatter = CustomJSTickFormatter(
        code=f"""
        return "";
    """
    )

    core_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color="black",
        fill_color="white",
    )
    core_renderer = plot.add_glyph(cores_data_source, core_glyph)

    buffer_page_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color=None,
        fill_color="color",
    )
    plot.add_glyph(buffer_pages_data_source, buffer_page_glyph)

    plot.add_tools(
        WheelZoomTool(),
        PanTool(),
        ResetTool(),
        ZoomInTool(),
        ZoomOutTool(),
        HoverTool(
            renderers=[core_renderer],
            tooltips=[("Core", "(@core_y, @core_x)")],
        ),
    )
    return components(plot)


@app.route("/operation_buffer_report/<operation_id>")
def operation_buffer_report(operation_id):
    summarized_report_script, summarized_report_div = create_summarized_l1_buffer_plot(operation_id)
    detailed_report_script, detailed_report_div = create_detailed_l1_buffer_plot(operation_id)

    return render_template(
        "buffer_report.html",
        summarized_report_script=summarized_report_script,
        summarized_report_div=summarized_report_div,
        detailed_report_script=detailed_report_script,
        detailed_report_div=detailed_report_div,
    )


@app.route("/operation_graph/<operation_id>")
def operation_graph(operation_id):
    svg_file = ttnn.REPORTS_PATH / "graphs" / f"{operation_id}.svg"
    if not svg_file.exists():
        return "Graph not found! Was TTNN_ENABLE_GRAPH_REPORT=True set?"
    with open(svg_file) as f:
        content = f.read()
    return content


@app.route("/operation_codegen/<operation_id>")
def operation_codegen(operation_id):
    # codegen_file = ttnn.REPORTS_PATH / "codegen" / f"{operation_id}.py"
    # with open(codegen_file) as f:
    #     content = f.read()
    # return render_template("operation_codegen.html", content=content)
    return "Feature not implemented yet!"


@app.route("/operation_tensor_report/<operation_id>")
def operation_tensor_report(operation_id):
    operation = query_operation_by_id(operation_id=operation_id)

    input_tensors_path = ttnn.REPORTS_PATH / "input_tensors" / f"{operation_id}"
    output_tensors_path = ttnn.REPORTS_PATH / "output_tensors" / f"{operation_id}"

    input_tensors = []
    for tensor_path in input_tensors_path.glob("*.bin"):
        input_tensors.append(ttnn.load_tensor(tensor_path))

    output_tensors = []
    for tensor_path in output_tensors_path.glob("*.bin"):
        output_tensors.append(ttnn.load_tensor(tensor_path))

    return render_template(
        "operation_tensor_report.html", operation=operation, input_tensors=input_tensors, output_tensors=output_tensors
    )


if __name__ == "__main__":
    app.run(debug=True)
