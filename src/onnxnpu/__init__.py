"""Top-level package for **ONNXNPU Toolkit**.

Exposes:
    • `__version__`        - resolved at runtime via importlib.metadata
    • `Checker`, `Report`  - core classes for programmatic use
    • helper functions (`load_profile`, `iter_profiles`, …)

Quick usage
-----------
>>> from onnxnpu import Checker, load_profile
>>> report = Checker('model.onnx', load_profile('kl720')).run()
>>> print(report)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # Resolve installed package version; fallback for editable mode
    __version__: str = version(__name__)  # onnxnpu
except PackageNotFoundError:  # pragma: no cover  – during local dev
    __version__ = "0.0.dev0"

# Re‑export public API for convenience ---------------------------------------

from .checker import (  # noqa: F401  (re‑export)
    Checker,
    Report,
    SYMBOLS,
    iter_profiles,
    load_profile,
    print_model_summary,
    print_summary,
)

from .optimizer import (  # noqa: F401  (re‑export)
    update_opset_version,
)

__all__ = [
    "__version__",
    "Checker",
    "Report",
    "SYMBOLS",
    "iter_profiles",
    "load_profile",
    "print_model_summary",
    "print_summary",
    "update_opset_version",
]