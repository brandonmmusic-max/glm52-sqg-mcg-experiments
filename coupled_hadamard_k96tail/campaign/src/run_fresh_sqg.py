"""Operator CLI for the isolated four-layer fresh-SQG experiment.

No subcommand launches another process.  ``plan-jobs`` only writes inert argv
records; the four GPU workers begin only if an operator explicitly runs them.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .fresh_pipeline_common import SELECTED_LAYERS
from .fresh_pipeline_runner import (
    PipelinePaths,
    PipelineSettings,
    build_job_plan,
    encode_layer,
    evaluate_holdout,
    open_layer_runtime,
    prepare_layer,
    run_layer,
    seal_run,
    search_profiles,
    write_preflight,
)


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-seal", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--bit-contract", required=True)
    parser.add_argument("--kquant-root", required=True)
    parser.add_argument("--exllamav3-root", required=True)
    parser.add_argument("--sqg-extension-seal", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)


def _worker(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--layer", type=int, choices=SELECTED_LAYERS, required=True)
    parser.add_argument("--device", default="cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh BF16+capture SQG four-layer treatment pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="seal read-only inputs")
    _paths(preflight)
    preflight.add_argument(
        "--skip-capture-payload-hashes",
        action="store_true",
        help="development-only; workers still recheck all capture hashes",
    )
    preflight.add_argument(
        "--skip-source-payload-hashes",
        action="store_true",
        help="reuse a freshly sealed BF16 manifest; bind headers and file identities",
    )
    plan = sub.add_parser("plan-jobs", help="write inert four-GPU argv records")
    plan.add_argument("--preflight", required=True)
    plan.add_argument("--python-executable", default="python3")
    for name in (
        "prepare-layer",
        "search-profiles",
        "encode-layer",
        "holdout-layer",
        "run-layer",
    ):
        _worker(sub.add_parser(name))
    seal = sub.add_parser("seal-run")
    seal.add_argument("--preflight", required=True)
    dry = sub.add_parser("dry-run", help="print protocol without reading or writing")
    dry.add_argument("--run-id", default="glm52-fresh-sqg")
    return parser


def _dump(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dry-run":
        _dump(
            {
                "run_id": args.run_id,
                "layers": list(SELECTED_LAYERS),
                "one_layer_per_gpu": True,
                "profile_factorial": {
                    "sign_draws": 8,
                    "families": [
                        "identity",
                        "aggregate_rms",
                        "quarter_rms",
                        "inverse_quarter_rms",
                    ],
                    "cells_per_layer": 32,
                    "proxy_pruning": False,
                },
                "stages": [
                    "preflight",
                    "fit_preparation",
                    "exact_profile_factorial_selection",
                    "all_expert_fresh_sqg_encode",
                    "topology_neutral_layer_assembly",
                    "report_only_holdout",
                    "four_layer_seal",
                ],
                "model_workload_launched": False,
                "production_container_touched": False,
                "existing_model_mutated": False,
                "runnable_model_materialized": False,
            }
        )
        return 0
    if args.command == "preflight":
        paths = PipelinePaths.resolve(
            source_seal=args.source_seal,
            capture_dir=args.capture_dir,
            bit_contract=args.bit_contract,
            kquant_root=args.kquant_root,
            exllamav3_root=args.exllamav3_root,
            sqg_extension_seal=args.sqg_extension_seal,
            output_root=args.output_root,
        )
        settings = PipelineSettings(run_id=args.run_id)
        path = write_preflight(
            paths,
            settings,
            verify_capture_hashes=not args.skip_capture_payload_hashes,
            verify_source_hashes=not args.skip_source_payload_hashes,
        )
        _dump({"preflight": str(path), "model_workload_launched": False})
        return 0
    if args.command == "plan-jobs":
        _dump(
            build_job_plan(
                args.preflight, python_executable=args.python_executable
            )
        )
        return 0
    if args.command == "seal-run":
        _dump(seal_run(args.preflight))
        return 0
    runtime = open_layer_runtime(
        args.preflight, layer=args.layer, device=args.device
    )
    operation = {
        "prepare-layer": prepare_layer,
        "search-profiles": search_profiles,
        "encode-layer": encode_layer,
        "holdout-layer": evaluate_holdout,
        "run-layer": run_layer,
    }[args.command]
    _dump(operation(runtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
