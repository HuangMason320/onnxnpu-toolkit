"""Core logic for **ONNXNPU Toolkit**.

This module exposes:
    * `load_profile()`            - read a hardware profile JSON
    * `print_model_summary()`     - show model IO & dynamic-axes status
    * `print_summary()`           - one-line compatibility recap
    * `Checker` / `Report`        - main scanning classes

It is intentionally free of CLI/argparse so it can be reused programmatically.
"""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import onnx  # type: ignore
except ImportError as e:  # pragma: no cover
    print("[ERROR] Missing dependency 'onnx'.  Run: pip install onnx", file=sys.stderr)
    raise e

__all__ = [
    "SYMBOLS",
    "load_profile",
    "iter_profiles",
    "print_model_summary",
    "print_summary",
    "Checker",
    "Report",
    "valid_check",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = {"supported": "✓", "partial": "△", "unsupported": "✗"}
_PROFILE_PACKAGE = "onnxnpu.profiles"  # package where json files live


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def iter_profiles() -> List[str]:
    """Return list of built-in profile names (without .json)."""
    files = resources.files(_PROFILE_PACKAGE)
    return sorted(p.stem for p in files.iterdir() if p.suffix == ".json")  # type: ignore


def load_profile(name: str | Path) -> Dict:
    """Load profile by *name* (kl720) or custom JSON *path*."""
    p = Path(name)
    if p.suffix == "":  # look up packaged profile
        res = resources.files(_PROFILE_PACKAGE) / f"{name.lower()}.json"  # type: ignore
        if not res.exists():
            raise FileNotFoundError(f"Profile not found: {name}")
        data = res.read_text(encoding="utf-8")
        return json.loads(data)

    # direct path provided
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())

# ---------------------------------------------------------------------------
# Model summary & dynamic‑axes detection
# ---------------------------------------------------------------------------

def _shape_to_list(tensor_type) -> List[str]:
    dims: List[str] = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)
        elif d.HasField("dim_value") and d.dim_value > 0:
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return dims


def _tensor_info(value_info) -> Tuple[str, str, List[str]]:
    name = value_info.name
    ttype = value_info.type.tensor_type
    np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(ttype.elem_type, "?")
    elem_type = np_type.__name__ if hasattr(np_type, "__name__") else str(np_type)
    shape = _shape_to_list(ttype)
    return name, elem_type, shape


def print_model_summary(model_path: Path) -> bool:
    """Print basic info; return True if dynamic axes present."""
    print()
    
    model = onnx.load(str(model_path))
    ir_version = model.ir_version
    opset_version = max(op.version for op in model.opset_import)
    
    section_line = "═" * 60
    print(section_line)
    print("MODEL INFO")
    print(section_line)

    print(f"Name - {model_path.name}")
    print(f"IR version : {ir_version}")
    print(f"Opset : {opset_version}")

    def _dump(title: str, items):
        print(f"{title:<8}:", end=" ")
        if not items:
            print("<none>")
            return
        first = True
        for v in items:
            name, dtype, shape = _tensor_info(v)
            line = f"{name}  {dtype}  [{', '.join(shape)}]"
            print(line if first else " " * 9 + line)
            first = False

    _dump("Inputs", model.graph.input)
    _dump("Outputs", model.graph.output)

    # detect dynamic dims
    dynamic = any(
        not d.isdigit()
        for vi in (*model.graph.input, *model.graph.output)
        for d in _tensor_info(vi)[2]
    )
    print(
        f"Dynamic axes : {'Detected ' + SYMBOLS['unsupported'] if dynamic else 'None detected ' + SYMBOLS['supported']}"
    )
    print()
    return dynamic


# ---------------------------------------------------------------------------
# Compatibility summary helper
# ---------------------------------------------------------------------------

def print_summary(report: "Report") -> None:
    op_total = sum(cnt for cnt, _, _ in report.info.values())
    unsupported = sum(1 for _, status, _ in report.info.values() if status == "unsupported")
    partial = sum(1 for _, status, _ in report.info.values() if status == "partial")

    if unsupported:
        symbol = SYMBOLS["unsupported"]
        msg = f"{unsupported} unsupported operator(s) detected"
    elif partial:
        symbol = SYMBOLS["partial"]
        msg = f"{partial} partially supported operator(s) detected"
    else:
        symbol = SYMBOLS["supported"]
        msg = "All operators are supported"
    print()
    print(f"Summary: {msg} on {report.hw_name} {symbol}")
    print(f"Total operators: {len(report.info)} (instances: {op_total})")

# ---------------------------------------------------------------------------
# Checker / Report
# ---------------------------------------------------------------------------

class Checker:
    """Compare an ONNX model's operators to a hardware profile."""

    def __init__(self, model: Path, profile: Dict):
        self.model_path = model
        self.profile = profile
        self.onnx_model = onnx.load(str(model))
        self.profile_ops = {k.lower(): v for k, v in profile.get("operators", {}).items()}

    def run(self) -> "Report":
        info: Dict[str, Tuple[int, str, str | None]] = {}
        for node in self.onnx_model.graph.node:
            op = node.op_type
            cnt, _, _ = info.get(op, (0, "supported", None))
            cnt += 1
            spec = self.profile_ops.get(op.lower())
            status = spec["status"] if spec else "unsupported"
            note = spec.get("constraints") if (spec and status != "supported") else None
            info[op] = (cnt, status, note)
        
        # Get size limits from profile
        usb_limit = None
        flash_limit = None
        compression_ratio = 0.25  # Default 4:1 ratio
        
        if "memory_limits" in self.profile:
            mem_limits = self.profile["memory_limits"]
            if "usb_model_size" in mem_limits:
                usb_limit = mem_limits["usb_model_size"]
            if "flash_model_size" in mem_limits:
                flash_limit = mem_limits["flash_model_size"]
            if "compression_ratio" in mem_limits:
                compression_ratio = mem_limits["compression_ratio"]
        
        # Estimate size if at least one limit is defined
        estimated_size = None
        if usb_limit or flash_limit:
            estimated_size = estimate_nef_size(self.model_path, compression_ratio)
        
        return Report(
            self.model_path.name, 
            self.onnx_model.ir_version, 
            info, 
            self.profile.get("name", "unknown"),
            estimated_size=estimated_size,
            usb_limit=usb_limit,
            flash_limit=flash_limit
        )


class Report:
    def __init__(self, model_name, ir_version, info, hw_name, 
                 estimated_size=None, usb_limit=None, flash_limit=None):
        self.model_name = model_name
        self.ir_version = ir_version
        self.info = info
        self.hw_name = hw_name
        self.estimated_size = estimated_size
        self.usb_limit = usb_limit
        self.flash_limit = flash_limit

    # ---------- plain text ----------
    def to_text(self) -> str:
        section_line = "═" * 60
        
        # Setup operator table
        rows = [("Status", "Operator", "Count", "Notes")]
        for op in sorted(self.info):
            cnt, status, note = self.info[op]
            rows.append((SYMBOLS.get(status, "?"), op, str(cnt), note or ""))
        
        # Calculate widths for each column
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        fmt = "| " + " | ".join([f"{{:^{widths[0]}}}"] + [f"{{:<{w}}}" for w in widths[1:]]) + " |"
        
        # Create horizontal border
        hborder = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        
        # Build output with sections
        out = [
            section_line,
            f"HARDWARE COMPATIBILITY - {self.hw_name.upper()}",
            section_line,
            hborder,
            fmt.format(*rows[0]),
            "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        ]
        
        # Add each data row
        for row in rows[1:]:
            out.append(fmt.format(*row))
        
        # Add bottom border
        out.append(hborder)
        
        # Add memory requirements section
        if self.estimated_size is not None:
            size_mb = self.estimated_size / (1024 * 1024)
            out.extend([
                "",
                section_line,
                f"MEMORY REQUIREMENTS - {self.hw_name.upper()}",
                section_line,
                f"Estimated NEF size:   {size_mb:.2f} MB"
            ])
            
            if self.usb_limit is not None:
                usb_mb = self.usb_limit / (1024 * 1024)
                status = "-> MIGHT EXCEEDS LIMIT" if self.estimated_size > self.usb_limit else "-> OK"
                out.append(f"USB model limit:      {usb_mb:.2f} MB  {status}")
                    
            if self.flash_limit is not None:
                flash_mb = self.flash_limit / (1024 * 1024)
                status = "-> MIGHT EXCEEDS LIMIT" if self.estimated_size > self.flash_limit else "-> OK"
                out.append(f"Flash model limit:    {flash_mb:.2f} MB  {status}")
        
        # Add summary section
        # op_total = sum(cnt for cnt, _, _ in self.info.values())
        # unsupported = sum(1 for _, status, _ in self.info.values() if status == "unsupported")
        # partial = sum(1 for _, status, _ in self.info.values() if status == "partial")
        
        # out.extend([
        #     "",
        #     section_line,
        #     "SUMMARY",
        #     section_line
        # ])
        
        # if unsupported:
        #     out.append(f"⚠️ Operators: {unsupported} unsupported operator(s) detected")
        # elif partial:
        #     out.append(f"△ Operators: {partial} partially supported operator(s) detected")
        # else:
        #     out.append(f"✓ Operators: All {len(self.info)} operators ({op_total} instances) are supported")
        
        # # Add size check summary if applicable
        # if self.estimated_size is not None and (self.usb_limit or self.flash_limit):
        #     # Check if any limit is exceeded
        #     exceeded = False
        #     if self.usb_limit and self.estimated_size > self.usb_limit:
        #         exceeded = True
        #         exceed_by = (self.estimated_size - self.usb_limit) / (1024 * 1024)
        #         out.append(f"⚠️ Memory: Exceeds USB limit by {exceed_by:.2f} MB")
            
        #     if self.flash_limit and self.estimated_size > self.flash_limit:
        #         exceeded = True
        #         exceed_by = (self.estimated_size - self.flash_limit) / (1024 * 1024)
        #         out.append(f"⚠️ Memory: Exceeds Flash limit by {exceed_by:.2f} MB")
            
        #     if not exceeded:
        #         out.append(f"✓ Memory: Within all hardware limits")
        
        return "\n".join(out)

    # ---------- markdown ----------
    def to_markdown(self) -> str:
        md = [
            f"### {self.model_name} · IR {self.ir_version} · **{self.hw_name.upper()}**\n",
            "| Status | Operator | Count | Notes |",
            "| ------ | -------- | ----- | ----- |",
        ]
        for op in sorted(self.info):
            cnt, status, note = self.info[op]
            md.append(f"| {SYMBOLS.get(status, '?')} | {op} | {cnt} | {note or ''} |")
        
        # Add size information
        if self.estimated_size is not None:
            size_mb = self.estimated_size / (1024 * 1024)
            md.append(f"\n\n### Model Size Estimate\n")
            md.append(f"**Estimated NEF size:** {size_mb:.2f} MB\n")
            
            md.append("| Load Method | Size Limit | Status |")
            md.append("| ----------- | ---------- | ------ |")
            
            if self.usb_limit is not None:
                usb_mb = self.usb_limit / (1024 * 1024)
                status = "❌ EXCEEDS LIMIT" if self.estimated_size > self.usb_limit else "✅ OK"
                md.append(f"| USB Upload | {usb_mb:.0f} MB | {status} |")
                
            if self.flash_limit is not None:
                flash_mb = self.flash_limit / (1024 * 1024)
                status = "❌ EXCEEDS LIMIT" if self.estimated_size > self.flash_limit else "✅ OK"
                md.append(f"| Flash Load | {flash_mb:.0f} MB | {status} |")
        
        return "\n".join(md)

    __str__ = to_text

# ---------------------------------------------------------------------------
# Check Availability of NEF
# ---------------------------------------------------------------------------

def estimate_nef_size(model_path: Path, compression_ratio: float = 0.25) -> int:
    """
    Estimate the size of the NEF file that would be generated from this ONNX model.
    Using the specified compression ratio (default 4:1).
    
    Args:
        model_path: Path to the ONNX model file
        compression_ratio: Expected compression ratio (default 0.25, representing 4:1)
        
    Returns:
        Estimated size in bytes
    """
    # Get raw ONNX model size
    onnx_size = model_path.stat().st_size
    
    # Apply compression ratio to estimate NEF size
    estimated_size = int(onnx_size * compression_ratio)
    
    return estimated_size

def valid_check(model_path: Path) -> bool:
    """
    Verify that an ONNX model is valid and return whether it has dynamic axes.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        bool: True if the model has dynamic axes, False if all dimensions are static
        
    Raises:
        onnx.checker.ValidationError: If the model is invalid
    """
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Use the existing dynamic axes detection from print_model_summary
        dynamic = any(
            not d.isdigit()
            for vi in (*model.graph.input, *model.graph.output)
            for d in _tensor_info(vi)[2]
        )
        
        return dynamic
        
    except onnx.checker.ValidationError as e:
        print(f"[ERROR] Invalid model: {model_path.name}")
        print(f"  {e}")
        raise