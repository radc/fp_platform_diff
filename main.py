"""Command-line entry point for the floating-point cross-platform experiment framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.generate_inputs import generate_inputs_from_config
from src.execute_ops import execute_from_config
from src.compare_runs import compare_runs


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        description="Floating-point cross-platform experiment framework"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate input tensors and save them to disk"
    )
    generate_parser.add_argument(
        "--config", type=Path, required=True, help="Path to the experiment JSON config"
    )

    execute_parser = subparsers.add_parser(
        "execute", help="Execute the operation script on a given device"
    )
    execute_parser.add_argument(
        "--config", type=Path, required=True, help="Path to the experiment JSON config"
    )
    execute_parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Execution device, for example: cpu or cuda:0",
    )
    execute_parser.add_argument(
        "--run-name", type=str, required=True, help="Name of the execution run"
    )
    execute_parser.add_argument(
        "--operation-file",
        type=Path,
        required=True,
        help="Path to the user-editable operation.py file",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare one reference execution folder against one or more candidate folders",
    )
    compare_parser.add_argument(
        "--reference", type=Path, required=True, help="Reference execution directory"
    )
    compare_parser.add_argument(
        "--candidate",
        type=Path,
        nargs="+",
        required=True,
        help="One or more candidate execution directories",
    )
    compare_parser.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=["pt", "bin", "txt"],
        help="File format used for step comparison",
    )
    compare_parser.add_argument(
        "--rtol", type=float, default=0.0, help="Relative tolerance"
    )
    compare_parser.add_argument(
        "--atol", type=float, default=0.0, help="Absolute tolerance"
    )

    return parser


def main() -> None:
    """Program entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        generate_inputs_from_config(args.config)
    elif args.command == "execute":
        execute_from_config(
            config_path=args.config,
            device=args.device,
            run_name=args.run_name,
            operation_file=args.operation_file,
        )
    elif args.command == "compare":
        compare_runs(
            reference_dir=args.reference,
            candidate_dirs=args.candidate,
            tensor_format=args.format,
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()