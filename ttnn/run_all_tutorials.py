import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def run_all_notebooks(directory_path):
    all_passed = True
    for filename in os.listdir(directory_path):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(directory_path, filename)
            with open(notebook_path) as f:
                notebook = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            try:
                ep.preprocess(notebook, {"metadata": {"path": directory_path}})
                print(f"Executed {notebook_path} successfully.")
            except CellExecutionError as e:
                all_passed = False
                print(f"Error occurred while executing {notebook_path}: {e}")
            except Exception as e:
                all_passed = False
                print(f"An unexpected error occurred while executing {notebook_path}: {e}")
    return all_passed


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_directory, "tutorials")
    assert run_all_notebooks(directory_path), "Unable to run all tutorials"
