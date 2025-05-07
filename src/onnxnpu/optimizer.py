"""Optimization functions for ONNX models in the ONNXNPU Toolkit.

This module contains functions for modifying and optimizing ONNX models:
    * `update_opset_version()` - modify model's opset version
"""

from __future__ import annotations

from pathlib import Path

try:
    import onnx  # type: ignore
except ImportError as e:  # pragma: no cover
    import sys
    print("[ERROR] Missing dependency 'onnx'.  Run: pip install onnx", file=sys.stderr)
    raise e

__all__ = [
    "update_opset_version",
]

def update_opset_version(model_path: Path, target_version: int, output_path: Path = None) -> Path:
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