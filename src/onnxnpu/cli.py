"""Command-line interface for **ONNXNPU Toolkit**.

After installing the package, this module is exposed as the console-scripts
`onpu` and `onnxnpu` (configured in pyproject.toml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from . import __version__
from .checker import (
    Checker,
    iter_profiles,
    load_profile,
    print_model_summary,
    print_summary,
)
from .optimizer import update_opset_version

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser with subcommands."""
    # Main parser
    parser = argparse.ArgumentParser(
        prog="onpu",
        description="ONNXNPU Toolkit - Check, optimize and modify ONNX models for NPU deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"{__version__}",
        help="Show version number and exit"
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Check command
    check_parser = subparsers.add_parser(
        "check", 
        help="Check ONNX model compatibility with NPU hardware profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    check_parser.add_argument("model", help="Path to .onnx model file")
    check_parser.add_argument(
        "-p",
        "--hardware",
        nargs="*",
        help="Profile name(s) (e.g. kl720) or JSON file path(s). Omit to scan all built-in profiles.",
    )
    check_parser.add_argument("--markdown", action="store_true", help="Output report(s) as Markdown")
    
    # Opt command
    opt_parser = subparsers.add_parser(
        "opt", 
        help="Optimize and modify ONNX models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opt_parser.add_argument("model", help="Path to .onnx model file")
    opt_parser.add_argument(
        "--opset", 
        type=int,
        help="Update model to specified opset version (1-18)",
        metavar="VERSION",
        choices=range(1, 19),  # Max version 18 as specified
    )
    opt_parser.add_argument(
        "-o", "--output",
        help="Output path for modified model",
    )
    
    return parser

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def check_command(args) -> None:
    """Handle the 'check' subcommand."""
    model_path = Path(args.model)
    
    # Show model IO + detect dynamic axes
    dynamic = print_model_summary(model_path)
    
    # Determine which profiles to scan
    profile_keys = args.hardware if args.hardware else iter_profiles()
    
    for key in profile_keys:
        profile = load_profile(key)
        
        # Warn if KL* profile & dynamic dims present
        if dynamic and profile.get("name", "").lower().startswith("kl"):
            print(f"[WARNING] Model uses dynamic axes which are NOT supported on {profile['name']}.")
            
        report = Checker(model_path, profile).run()
        print(report.to_markdown() if args.markdown else report)
        print_summary(report)
        print()

def opt_command(args) -> None:
    """Handle the 'opt' subcommand."""
    model_path = Path(args.model)
    
    # Handle opset update if requested
    if args.opset is not None:
        output_path = Path(args.output) if args.output else None
        updated_path = update_opset_version(model_path, args.opset, output_path)
        print_model_summary(updated_path)
    else:
        print("No optimization specified. Use --opset to update the opset version.")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    
    # Handle legacy command style (no subcommand)
    if args.command is None:
        # If the first argument looks like a file path, assume it's the 'check' command
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and Path(sys.argv[1]).exists():
            print("[DEPRECATED] Running in legacy mode. Please use 'onpu check' instead.")
            # Reconstruct arguments as if 'check' was specified
            if argv is None:
                argv = sys.argv[1:]
            else:
                argv = ['check'] + argv
            args = _build_parser().parse_args(argv)
        else:
            print("Error: No command specified. Use 'check' or 'opt'.")
            _build_parser().print_help()
            return
    
    # Dispatch to appropriate command handler
    if args.command == "check":
        check_command(args)
    elif args.command == "opt":
        opt_command(args)

if __name__ == "__main__":  # pragma: no cover
    main()