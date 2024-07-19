1. There are 7 conv's used in klassify model, all convs are failing due to OOM issue
2. To test the unit_test, run `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_klassify_new_conv`

Note: For conv checking purpose batch_size=1 is used even though 32 is used in the model.
