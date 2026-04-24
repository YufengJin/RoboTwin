#!/usr/bin/env python3
"""RoboTwin single-task demo rollout (WebSocket client).

Purpose:
    Connects to a ``policy_websocket`` policy server, runs a small number of
    demo episodes on a single RoboTwin task, mirrors stdout to
    ``<log_dir>/demo/<task>--<YYYYMMDD_HHMMSS>/demo.log``, and (with
    ``--save_video``) saves a ``demo.mp4`` side-by-side camera stitch for the
    first episode. This is a thin wrapper around ``script/run_demo_ws.py`` that
    applies the benchmark-env-generator contract (``--log_dir`` default
    ``./logs``, ``Purpose``/``Example`` docstring, canonical artifact naming);
    the heavy lifting lives in the upstream file, which is preserved intact.

Example:
    From the RoboTwin repo root (inside the robotwin-headless container)::

        python scripts/run_demo.py \\
            --task_name beat_block_hammer --task_config demo_clean \\
            --policy_server_addr localhost:8000 \\
            --num_resets 1 --log_dir ./logs --save_video

    Artefacts land in ``./logs/demo/beat_block_hammer--<ts>/``:
    ``demo.log`` (stdout tee) and, if ``--save_video`` is set, ``demo.mp4``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import runpy
import sys
from contextlib import contextmanager
from datetime import datetime


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_UPSTREAM_SCRIPT = os.path.join(_REPO_ROOT, "script", "run_demo_ws.py")
_UPSTREAM_UTILS_DIR = os.path.join(_REPO_ROOT, "script")


def _list_tasks() -> list:
    if _UPSTREAM_UTILS_DIR not in sys.path:
        sys.path.insert(0, _UPSTREAM_UTILS_DIR)
    from robotwin_run_utils import list_robotwin_task_names
    return list_robotwin_task_names()


def _parse_wrapper_args():
    tasks = _list_tasks()
    parser = argparse.ArgumentParser(
        description="RoboTwin demo client (benchmark-env-generator contract wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000")
    parser.add_argument("--policy", type=str, default="websocketPolicy")
    parser.add_argument("--num_resets", type=int, default=1,
                        help="Successful policy episodes to run (default 1 for smoke tests).")
    parser.add_argument("--instruction_type", type=str, default="unseen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_type", type=str, default="qpos",
                        choices=("qpos", "ee", "delta_ee"))
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Base log dir; demo artefacts land under <log_dir>/demo/<task>--<ts>/.")
    parser.add_argument("--save_video", action="store_true", default=False,
                        help="Save demo.mp4 (side-by-side camera stitch) in the run dir.")
    parser.add_argument("--list_tasks", action="store_true")
    args = parser.parse_args()
    if not args.list_tasks:
        if not args.task_name:
            parser.error("--task_name is required unless --list_tasks")
        if args.task_name not in tasks:
            parser.error(f"Unknown task_name {args.task_name!r}. Use --list_tasks.")
    return args


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


@contextmanager
def _tee_stdout(log_path: str):
    log_file = open(log_path, "w")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    try:
        yield log_file
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            log_file.close()
        except Exception:
            pass


def _canonicalize_demo_artifacts(run_dir: str) -> None:
    """Rename the first episode mp4 to demo.mp4 per contract."""
    for mp4 in sorted(glob.glob(os.path.join(run_dir, "episode=*.mp4"))):
        target = os.path.join(run_dir, "demo.mp4")
        if os.path.exists(target):
            os.remove(target)
        os.rename(mp4, target)
        break


def main() -> int:
    args = _parse_wrapper_args()

    if args.list_tasks:
        upstream_argv = [_UPSTREAM_SCRIPT, "--list_tasks"]
        sys.argv = upstream_argv
        runpy.run_path(_UPSTREAM_SCRIPT, run_name="__main__")
        return 0

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_log_dir = os.path.join(args.log_dir, "demo")
    run_dir = os.path.join(demo_log_dir, f"{args.task_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "demo.log")

    upstream_argv = [
        _UPSTREAM_SCRIPT,
        "--task_name", args.task_name,
        "--task_config", args.task_config,
        "--policy_server_addr", args.policy_server_addr,
        "--policy", args.policy,
        "--num_resets", str(args.num_resets),
        "--instruction_type", args.instruction_type,
        "--seed", str(args.seed),
        "--action_type", args.action_type,
        "--demo_log_dir", demo_log_dir,
    ]
    sys.argv = upstream_argv

    rc = 0
    with _tee_stdout(log_path) as log_file:
        log_file.write("# RoboTwin demo wrapper args\n")
        for k, v in sorted(vars(args).items()):
            log_file.write(f"#   {k} = {v}\n")
        log_file.write(f"# upstream script: {_UPSTREAM_SCRIPT}\n")
        log_file.write(f"# upstream argv : {upstream_argv}\n")
        log_file.write(f"# run_dir: {run_dir}\n")
        log_file.flush()

        try:
            runpy.run_path(_UPSTREAM_SCRIPT, run_name="__main__")
        except SystemExit as e:
            rc = int(e.code or 0)
        except Exception as e:
            log_file.write(f"\n# UPSTREAM RAISED: {type(e).__name__}: {e}\n")
            raise

        try:
            upstream_run_dir = os.path.join(demo_log_dir, f"{args.task_name}--{date_str}")
            if upstream_run_dir != run_dir and os.path.isdir(upstream_run_dir):
                pass
            if args.save_video:
                _canonicalize_demo_artifacts(run_dir)
            else:
                for mp4 in glob.glob(os.path.join(run_dir, "episode=*.mp4")):
                    os.remove(mp4)
        except Exception as e:
            log_file.write(f"\n# artefact post-processing failed: {type(e).__name__}: {e}\n")

    print(f"Run directory: {run_dir}")
    print(f"Log saved to:  {log_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
