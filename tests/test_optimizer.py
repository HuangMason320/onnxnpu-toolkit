import pytest
import onnx
import tempfile
import os
from pathlib import Path

from onnxnpu.optimizer import update_opset_version, optimize_model, infer_shapes


def _make_static_model(opset_version=13):
    node = onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'])
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    graph = onnx.helper.make_graph([node], 'test-model', [x], [y])
    model = onnx.helper.make_model(graph, producer_name='onnxnpu-test')
    model.opset_import[0].version = opset_version
    return model


def _save_model(model):
    f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    onnx.save(model, f.name)
    f.close()
    return Path(f.name)


def test_update_opset_version_upgrade(tmp_path):
    model_path = _save_model(_make_static_model(opset_version=12))
    output_path = tmp_path / "upgraded.onnx"
    try:
        result = update_opset_version(model_path, 14, output_path)
        assert result == output_path
        updated = onnx.load(str(output_path))
        versions = {op.domain: op.version for op in updated.opset_import}
        assert versions[""] == 14
    finally:
        os.unlink(model_path)


def test_update_opset_version_same(tmp_path):
    model_path = _save_model(_make_static_model(opset_version=13))
    try:
        result = update_opset_version(model_path, 13)
        assert result == model_path
    finally:
        os.unlink(model_path)


def test_update_opset_version_default_output():
    model_path = _save_model(_make_static_model(opset_version=12))
    try:
        result = update_opset_version(model_path, 14)
        assert result.exists()
        assert "opset14" in result.name
        os.unlink(result)
    finally:
        os.unlink(model_path)


def test_infer_shapes():
    model_path = _save_model(_make_static_model())
    try:
        result = infer_shapes(model_path)
        assert isinstance(result, onnx.ModelProto)
    finally:
        os.unlink(model_path)


def test_optimize_model_basic(tmp_path):
    model_path = _save_model(_make_static_model())
    output_path = tmp_path / "optimized.onnx"
    try:
        result_path, success = optimize_model(model_path, output_path, check_n=0)
        assert result_path == output_path
        assert output_path.exists()
    finally:
        os.unlink(model_path)


def test_optimize_model_default_output():
    model_path = _save_model(_make_static_model())
    try:
        result_path, success = optimize_model(model_path, check_n=0)
        assert result_path.exists()
        assert "_opt" in result_path.name
        os.unlink(result_path)
    finally:
        os.unlink(model_path)


def test_optimize_model_with_opset(tmp_path):
    model_path = _save_model(_make_static_model(opset_version=12))
    output_path = tmp_path / "optimized.onnx"
    try:
        result_path, _ = optimize_model(model_path, output_path, check_n=0, target_opset=14)
        assert output_path.exists()
        updated = onnx.load(str(output_path))
        versions = {op.domain: op.version for op in updated.opset_import}
        assert versions[""] == 14
    finally:
        os.unlink(model_path)
