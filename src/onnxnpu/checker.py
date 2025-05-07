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

    print(f"Model summary - {model_path.name}")
    print(f"IR version : {ir_version}    Opset : {opset_version}")

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
        return Report(self.model_path.name, self.onnx_model.ir_version, info, self.profile.get("name", "unknown"))


class Report:
    def __init__(self, model_name: str, ir_version: int, info: Dict[str, Tuple[int, str, str | None]], hw_name: str):
        self.model_name = model_name
        self.ir_version = ir_version
        self.info = info
        self.hw_name = hw_name

    # ---------- plain text ----------
    def to_text(self) -> str:
        rows = [("Status", "Operator", "Count", "Notes")]
        for op in sorted(self.info):
            cnt, status, note = self.info[op]
            rows.append((SYMBOLS.get(status, "?"), op, str(cnt), note or ""))
        
        # Calculate widths for each column
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        fmt = "| " + " | ".join([f"{{:^{widths[0]}}}"] + [f"{{:<{w}}}" for w in widths[1:]]) + " |"
        
        # Create horizontal border
        hborder = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        
        # Build output with box only around the table data
        out = [
            f"{self.model_name} · IR {self.ir_version} · {self.hw_name.upper()}",
            "",
            hborder,
            fmt.format(*rows[0]),
            "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        ]
        
        # Add each data row
        for row in rows[1:]:
            out.append(fmt.format(*row))
        
        # Add bottom border
        out.append(hborder)
        
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
        return "\n".join(md)

    __str__ = to_text
