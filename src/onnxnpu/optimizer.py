"""Optimization functions for ONNX models in the ONNXNPU Toolkit.

This module contains functions for modifying and optimizing ONNX models:
    * `update_opset_version()` - modify model's opset version
    * `optimize_model()` - simplify ONNX model using onnxsim
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

try:
    import onnx
    import onnx.shape_inference
except ImportError as e:  # pragma: no cover
    print("[ERROR] Missing dependency 'onnx'. Run: pip install onnx", file=sys.stderr)
    raise e

try:
    from onnxsim import simplify
except ImportError as e:  # pragma: no cover
    print("[ERROR] Missing dependency 'onnxsim'. Run: pip install onnxsim", file=sys.stderr)
    raise e

from .checker import valid_check

__all__ = [
    "update_opset_version",
    "optimize_model",
    "infer_shapes",
]

def infer_shapes(model_path: Path) -> onnx.ModelProto:
    """Infer shapes for all tensors in the model.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        Model with inferred shapes
    """
    model = onnx.load(str(model_path))
    model_with_shapes = onnx.shape_inference.infer_shapes(model)
    return model_with_shapes

def update_opset_version(model_path: Path, target_version: int, output_path: Optional[Path] = None) -> Path:
    """Update model's opset version to target_version.
    
    Args:
        model_path: Path to the original .onnx model
        target_version: Target opset version (integer)
        output_path: Path to save the modified model (defaults to original with suffix)
    
    Returns:
        Path to the saved modified model
    """
    model = onnx.load(str(model_path))
    
    # Check current versions
    current_versions = {op.domain: op.version for op in model.opset_import}
    default_domain_version = current_versions.get("", None)
    
    if default_domain_version is None:
        # If no default domain found, create one
        default_op = onnx.OperatorSetIdProto()
        default_op.domain = ""  # Default domain
        default_op.version = target_version
        model.opset_import.append(default_op)
        print(f"Added default opset version {target_version}")
    else:
        # Update default domain version
        for imp in model.opset_import:
            if imp.domain == "":  # Default domain
                if imp.version > target_version:
                    print(f"WARNING: Downgrading opset from {imp.version} to {target_version} may cause compatibility issues!")
                elif imp.version < target_version:
                    print(f"Upgrading opset from {imp.version} to {target_version}")
                else:
                    print(f"Model already at opset version {target_version}")
                    return model_path  # No changes needed
                
                imp.version = target_version
    
    # If no output path provided, create one based on input
    if output_path is None:
        stem = model_path.stem
        output_path = model_path.with_name(f"{stem}_opset{target_version}.onnx")
    
    # Save the modified model
    onnx.save(model, str(output_path))
    print(f"Model saved to: {output_path}")
    return output_path

# In optimizer.py, modify the optimize_model function:
def optimize_model(model_path: Path, output_path: Optional[Path] = None, 
                   check_n: int = 1, overwrite_input_shapes=None, 
                   skip_optimizers: Optional[list] = None,
                   target_opset: Optional[int] = None,
                   hardware_profile: Optional[str] = None) -> Tuple[Path, bool]:
    """Simplify ONNX model using onnxsim.
    
    Args:
        model_path: Path to the ONNX model file
        output_path: Path to save the optimized model (defaults to original with "_opt" suffix)
        check_n: Number of test runs for correctness verification (0 to skip checking)
        overwrite_input_shapes: Whether to overwrite input shapes with ones inferred
        skip_optimizers: List of optimizers to skip (None to use all)
        target_opset: If provided, update model to this opset version before optimization
        
    Returns:
        Tuple of (output_path, success)
    """
    temp_path = model_path
    
    # First, update opset if requested
    if target_opset is not None:
        print(f"Updating opset version to {target_opset}...")
        temp_dir = Path(output_path).parent if output_path else model_path.parent
        temp_file = temp_dir / f"{model_path.stem}_temp.onnx"
        temp_path = update_opset_version(model_path, target_opset, temp_file)
    
    print(f"Optimizing model: {temp_path}")
    model = onnx.load(str(temp_path))
    
    # Run shape inference
    try:
        print("Running ONNX shape inference...")
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Continuing with original model...")
    
    # Run onnxsim simplification
    try:
        print("Simplifying model with onnxsim...")
        # Convert boolean flag to the expected dictionary format if it's a boolean
        actual_shapes = {} if overwrite_input_shapes is True else None
        
        model_opt, check_ok = simplify(
            model,
            check_n=check_n,
            perform_optimization=True,
            skip_shape_inference=False,
            skipped_optimizers=skip_optimizers,
            overwrite_input_shapes=actual_shapes,  # Use the converted value
            include_subgraph=True,
        )
    
        if not check_ok and check_n > 0:
            print("WARNING: Model simplification might affect the model accuracy!")
    except Exception as e:
        print(f"Error during model simplification: {e}")
        model_opt = model
        check_ok = False
        
    # If no output path provided, create one based on input
    if output_path is None:
        stem = model_path.stem
        suffix = f"_opset{target_opset}_opt" if target_opset else "_opt"
        output_path = model_path.with_name(f"{stem}{suffix}.onnx")
    
    # Save the modified model
    onnx.save(model_opt, str(output_path))
    print(f"Optimized model saved to: {output_path}")
    
    # Clean up temporary file if created
    if temp_path != model_path and temp_path.exists():
        temp_path.unlink()
    
    # Validate the output model
    try:
        is_dynamic = valid_check(output_path)
        if is_dynamic:
            print("WARNING: Optimized model contains dynamic axes which may not be supported by Kneron NPUs")
    except Exception as e:
        print(f"WARNING: Validation failed for optimized model: {e}")
        check_ok = False
    
    # Perform hardware compatibility check if profile is provided
    if hardware_profile:
        from .checker import Checker, load_profile, print_summary, print_model_summary
        
        print(f"\nChecking optimized model compatibility with {hardware_profile}...")
        try:
            profile = load_profile(hardware_profile)
            checker = Checker(output_path, profile)
            report = checker.run()
            
            print_model_summary(output_path)
            
            print(report)
            print_summary(report)
            
            # Check for any unsupported operators
            unsupported = any(status == "unsupported" 
                              for _, status, _ in report.info.values())
            if unsupported:
                print("\nWARNING: Optimized model contains unsupported operators!")
                check_ok = False
                
            # Check for shape issues
            if report.shape_issues:
                print("\nWARNING: Optimized model has input shape issues!")
                check_ok = False
                
        except Exception as e:
            print(f"ERROR: Hardware compatibility check failed: {e}")
    
    return output_path, check_ok