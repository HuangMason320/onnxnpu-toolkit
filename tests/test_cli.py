import pytest
import onnx
import tempfile
import os
from pathlib import Path

from onnxnpu.cli import _build_parser, main


def _make_static_model():
    node = onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'])
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    graph = onnx.helper.make_graph([node], 'test-model', [x], [y])
    return onnx.helper.make_model(graph, producer_name='onnxnpu-test')


def _save_model(model):
    f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    onnx.save(model, f.name)
    f.close()
    return Path(f.name)


def test_parser_check_subcommand():
    parser = _build_parser()
    args = parser.parse_args(['check', 'model.onnx'])
    assert args.command == 'check'
    assert args.model == 'model.onnx'


def test_parser_opt_subcommand():
    parser = _build_parser()
    args = parser.parse_args(['opt', 'model.onnx', '--opset', '13'])
    assert args.command == 'opt'
    assert args.opset == 13


def test_parser_opt_opset_18():
    parser = _build_parser()
    args = parser.parse_args(['opt', 'model.onnx', '--opset', '18'])
    assert args.opset == 18


def test_parser_list_subcommand():
    parser = _build_parser()
    args = parser.parse_args(['list'])
    assert args.command == 'list'


def test_main_no_command(capsys):
    main([])
    captured = capsys.readouterr()
    assert "No command specified" in captured.out


def test_main_check(capsys):
    model_path = _save_model(_make_static_model())
    try:
        main(['check', str(model_path)])
        captured = capsys.readouterr()
        assert "MODEL INFO" in captured.out
    finally:
        os.unlink(model_path)


def test_main_list(capsys):
    main(['list'])
    captured = capsys.readouterr()
    assert "Available hardware profiles" in captured.out


def test_main_list_verbose(capsys):
    main(['list', '--verbose'])
    captured = capsys.readouterr()
    assert "Supported operators:" in captured.out
