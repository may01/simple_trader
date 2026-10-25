"""Provenance stamp shared by every artifact the harness writes."""
from __future__ import annotations

import datetime as dt
import os
import subprocess

SPEC = "external/docs/superpowers/experiment/swing_sinusoid.md"
PLAN = "external/docs/superpowers/plans/2026-10-24-swing-sinusoid.md"


def git_sha() -> str:
    env = os.environ.get("GIT_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 — container may lack git / safe.directory
        return "unknown"


def stamp(dataset: str, **extra) -> dict:
    return {
        "dataset": dataset,
        "git_sha": git_sha(),
        "spec": SPEC,
        "plan": PLAN,
        "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
