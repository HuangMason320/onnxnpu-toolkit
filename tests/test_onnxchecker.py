import pytest
import onnx
import tempfile
import os
from pathlib import Path

from onnxnpu.checker import (
    valid_check,
    has_dynamic_axes,
    load_profile,
    iter_profiles,
    Checker,
    estimate_nef_size,
    check_input_shape_constraints,
    print_model_summary,
    print_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_static_model():
    node = onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'])
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    graph = onnx.helper.make_graph([node], 'static-model', [x], [y])
    return onnx.helper.make_model(graph, producer_name='onnxnpu-test')


def _make_dynamic_model():
    node = onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'])
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ['batch', 3, 224, 224])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, ['batch', 3, 224, 224])
    graph = onnx.helper.make_graph([node], 'dynamic-model', [x], [y])
    return onnx.helper.make_model(graph, producer_name='onnxnpu-test')


def _save_model(model):
    f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    onnx.save(model, f.name)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# valid_check
# ---------------------------------------------------------------------------

def test_valid_model():
    model_path = _save_model(_make_static_model())
    try:
        assert valid_check(model_path) is False
    finally:
        os.unlink(model_path)


def test_valid_model_dynamic():
    model_path = _save_model(_make_dynamic_model())
    try:
        assert valid_check(model_path) is True
    finally:
        os.unlink(model_path)


def test_invalid_model():
    node = onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'])
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, None)
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    graph = onnx.helper.make_graph([node], 'invalid-model', [x], [y])
    model = onnx.helper.make_model(graph, producer_name='onnxnpu-test')

    model_path = _save_model(model)
    try:
        with pytest.raises(onnx.checker.ValidationError):
            valid_check(model_path)
    finally:
        os.unlink(model_path)


# ---------------------------------------------------------------------------
# has_dynamic_axes
# ---------------------------------------------------------------------------

def test_has_dynamic_axes_static():
    assert has_dynamic_axes(_make_static_model()) is False


def test_has_dynamic_axes_dynamic():
    assert has_dynamic_axes(_make_dynamic_model()) is True


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def test_iter_profiles_returns_list():
    profiles = iter_profiles()
    assert isinstance(profiles, list)
    assert len(profiles) > 0


def test_load_profile_by_name():
    profiles = iter_profiles()
    profile = load_profile(profiles[0])
    assert "name" in profile
    assert "operators" in profile


def test_load_profile_not_found():
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent_profile_xyz")


def test_load_profile_from_path(tmp_path):
    import json
    p = tmp_path / "custom.json"
    p.write_text(json.dumps({"name": "custom", "operators": {}}))
    profile = load_profile(str(p))
    assert profile["name"] == "custom"


# ---------------------------------------------------------------------------
# Checker / Report
# ---------------------------------------------------------------------------

def test_checker_run_static():
    model_path = _save_model(_make_static_model())
    try:
        profile = load_profile(iter_profiles()[0])
        checker = Checker(model_path, profile)
        report = checker.run()
        assert report.hw_name == profile["name"]
        assert "Relu" in report.info
    finally:
        os.unlink(model_path)


def test_checker_accepts_preloaded_model():
    model = _make_static_model()
    model_path = _save_model(model)
    try:
        profile = load_profile(iter_profiles()[0])
        checker = Checker(model_path, profile, onnx_model=model)
        report = checker.run()
        assert "Relu" in report.info
    finally:
        os.unlink(model_path)


def test_report_to_text():
    model_path = _save_model(_make_static_model())
    try:
        profile = load_profile(iter_profiles()[0])
        report = Checker(model_path, profile).run()
        text = report.to_text()
        assert "HARDWARE COMPATIBILITY" in text
        assert "Relu" in text
    finally:
        os.unlink(model_path)


def test_report_to_markdown():
    model_path = _save_model(_make_static_model())
    try:
        profile = load_profile(iter_profiles()[0])
        report = Checker(model_path, profile).run()
        md = report.to_markdown()
        assert "| Status |" in md
        assert "Relu" in md
    finally:
        os.unlink(model_path)


# ---------------------------------------------------------------------------
# estimate_nef_size
# ---------------------------------------------------------------------------

def test_estimate_nef_size():
    model_path = _save_model(_make_static_model())
    try:
        file_size = model_path.stat().st_size
        estimated = estimate_nef_size(model_path, compression_ratio=0.25)
        assert estimated == int(file_size * 0.25)
    finally:
        os.unlink(model_path)


# ---------------------------------------------------------------------------
# check_input_shape_constraints
# ---------------------------------------------------------------------------

def test_check_input_shape_constraints_valid():
    model = _make_static_model()
    issues = check_input_shape_constraints(model)
    assert len(issues) == 0


def test_check_input_shape_constraints_dynamic():
    model = _make_dynamic_model()
    issues = check_input_shape_constraints(model)
    assert len(issues) > 0
    issue = list(issues.values())[0]
    assert len(issue["issues"]) > 0


def test_check_input_shape_constraints_wrong_batch():
    node = onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'])
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [4, 3, 224, 224])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [4, 3, 224, 224])
    graph = onnx.helper.make_graph([node], 'batch4-model', [x], [y])
    model = onnx.helper.make_model(graph, producer_name='onnxnpu-test')
    issues = check_input_shape_constraints(model)
    assert len(issues) > 0
    assert any("Batch dimension" in i for i in list(issues.values())[0]["issues"])


# ---------------------------------------------------------------------------
# print_model_summary / print_summary (smoke tests)
# ---------------------------------------------------------------------------

def test_print_model_summary(capsys):
    model_path = _save_model(_make_static_model())
    try:
        dynamic = print_model_summary(model_path)
        assert dynamic is False
        captured = capsys.readouterr()
        assert "MODEL INFO" in captured.out
    finally:
        os.unlink(model_path)


def test_print_summary(capsys):
    model_path = _save_model(_make_static_model())
    try:
        profile = load_profile(iter_profiles()[0])
        report = Checker(model_path, profile).run()
        print_summary(report)
        captured = capsys.readouterr()
        assert "Summary:" in captured.out
    finally:
        os.unlink(model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
