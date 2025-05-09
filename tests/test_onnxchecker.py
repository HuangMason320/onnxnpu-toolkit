import pytest
import onnx
import tempfile
import os
from pathlib import Path

def valid_check(model_path):
    """ 
    Check if the model is valid and return whether it has dynamic axes.
    Returns True if the model has dynamic axes, False if all dimensions are static.
    """
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Check for dynamic dimensions in inputs and outputs
        dynamic = False
        for tensor in list(model.graph.input) + list(model.graph.output):
            shape = tensor.type.tensor_type.shape
            for dim in shape.dim:
                # If dim has dim_param or no dim_value, it's dynamic
                if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                    dynamic = True
                    break
            if dynamic:
                break
                
        return dynamic
        
    except onnx.checker.ValidationError as e:
        print(f"[ERROR] Invalid model: {model_path.name}")
        print(f"  {e}")
        raise e

def test_valid_model():
    """
    Test that valid ONNX models pass through check_model and valid_check correctly.
    This test creates a simple ONNX model with a single Relu node, saves it to a temporary file,
    and verifies that valid_check correctly identifies it as a non-dynamic model.
    The function performs the following steps:
    1. Creates a Relu operation node
    2. Defines input and output tensors with fixed shapes
    3. Builds a graph with the node
    4. Creates an ONNX model from the graph
    5. Saves the model to a temporary file
    6. Calls valid_check on the model file
    7. Verifies that the model is identified as non-dynamic (static)
    8. Cleans up the temporary file
    Assertions:
        - The model should be identified as non-dynamic (dynamic is False)
    """
    node = onnx.helper.make_node(
        'Relu',
        inputs=['x'],
        outputs=['y'],
    )
    
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    
    graph = onnx.helper.make_graph(
        [node],
        'test-model',
        [x],
        [y],
    )
    
    model = onnx.helper.make_model(graph, producer_name='onnxnpu-test')
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = Path(f.name)
    
    try:
        dynamic = valid_check(model_path)
        assert dynamic is False 
    finally:
        os.unlink(model_path)

def test_invalid_model():
    """
    Test that invalid models correctly raise exceptions.
    This test function creates an invalid ONNX model with a Relu node,
    where the input tensor doesn't have specified dimensions which
    contradicts with the output tensor that has a fixed shape.
    The function then:
    1. Creates a temporary ONNX model file
    2. Saves the invalid model to the temporary file
    3. Verifies that a ValidationError is raised when calling valid_check()
    4. Cleans up by removing the temporary file
    Raises:
        AssertionError: If ValidationError is not raised when using the invalid model
    Note:
        The test uses pytest's raises context manager to verify the correct exception is raised.
    """
    node = onnx.helper.make_node(
        'Relu',
        inputs=['x'],
        outputs=['y'],
    )
    
    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, None)
    y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    
    graph = onnx.helper.make_graph(
        [node],
        'invalid-model',
        [x],
        [y],
    )
    
    model = onnx.helper.make_model(graph, producer_name='onnxnpu-test')
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = Path(f.name)
    
    try:
        with pytest.raises(onnx.checker.ValidationError):
            valid_check(model_path)
    finally:
        os.unlink(model_path)
        
if __name__ == "__main__":
    test_valid_model()
    test_invalid_model()