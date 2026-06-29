"""nn/orchestration/cli.py — thin argparse wrapper over Layer-B helpers.

`python -m nn.orchestration.cli <cmd>`

Three subcommands (the deterministic hands the skill calls so the agent never
re-implements path math, Docker launch, or promote/revert arithmetic freehand):

  materialize-spec --spec-yaml <path-in> --pair <pair>
      → prints the on-volume materialized spec path (one line)

  run-version --spec-path <vol path> --study <name> --pair <pair> --timeout <s>
      → prints JSON {"study": ..., "holdout_score": ...}

  decide --best <f|none> --candidate <f> --margin <f> --strike <i> --k <i>
         --version-n <i> --max-versions <i> --elapsed <f> --budget <f>
      → prints JSON of the LineageDecision

All helpers are imported at module top so unit tests can monkeypatch them on
the `cli` module object (e.g. `monkeypatch.setattr(cli, "decide", fake)`).
"""

import argparse
import json

from nn.orchestration.spec_store import materialize_spec
from nn.orchestration.runner import run_version_training, VersionResult
from nn.orchestration.lineage import decide, LineageDecision
from nn.nn_model_spec import NNModelSpec
from nn.device import nn_artefact_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nn.orchestration.cli")
    sub = parser.add_subparsers(dest="cmd")

    p_mat = sub.add_parser("materialize-spec")
    p_mat.add_argument("--spec-yaml", required=True)
    p_mat.add_argument("--pair", required=True)

    p_run = sub.add_parser("run-version")
    p_run.add_argument("--spec-path", required=True)
    p_run.add_argument("--study", required=True)
    p_run.add_argument("--pair", required=True)
    p_run.add_argument("--timeout", type=float, required=True)

    p_dec = sub.add_parser("decide")
    p_dec.add_argument("--best", required=True)            # "none" or a float string
    p_dec.add_argument("--candidate", type=float, required=True)
    p_dec.add_argument("--margin", type=float, required=True)
    p_dec.add_argument("--strike", type=int, required=True)
    p_dec.add_argument("--k", type=int, required=True)
    p_dec.add_argument("--version-n", type=int, required=True)
    p_dec.add_argument("--max-versions", type=int, required=True)
    p_dec.add_argument("--elapsed", type=float, required=True)
    p_dec.add_argument("--budget", type=float, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "materialize-spec":
        spec = NNModelSpec.from_yaml(args.spec_yaml)
        path = materialize_spec(spec, nn_artefact_root(args.pair))
        print(path)
        return 0

    if args.cmd == "run-version":
        r = run_version_training(
            args.spec_path, args.study, pair=args.pair, timeout_s=args.timeout,
        )
        print(json.dumps({"study": r.study, "holdout_score": r.holdout_score}))
        return 0

    if args.cmd == "decide":
        best = None if args.best == "none" else float(args.best)
        d = decide(
            best_score=best,
            candidate_score=args.candidate,
            margin=args.margin,
            strike_count=args.strike,
            K=args.k,
            version_n=args.version_n,
            max_versions=args.max_versions,
            elapsed_s=args.elapsed,
            budget_s=args.budget,
        )
        print(json.dumps({
            "action": d.action,
            "new_best": d.new_best,
            "strike_count": d.strike_count,
            "stop": d.stop,
            "reason": d.reason,
        }))
        return 0

    parser.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
