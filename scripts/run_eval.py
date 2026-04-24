#!/usr/bin/env python3
"""RoboTwin multi-episode evaluation (WebSocket client).

Purpose:
    Runs N evaluation episodes against a running ``policy_websocket`` policy
    server, aggregates success rate + avg episode length, writes a rolling-rate
    + FINAL RESULTS eval log to ``<log_dir>/eval/<task>--<YYYYMMDD_HHMMSS>/eval.log``,
    and (with ``--save_video``) saves one MP4 per episode named
    ``episode_{ep:03d}_success={bool}.mp4`` per the benchmark-env-generator
    contract. Thin wrapper around ``script/run_eval_ws.py``; adds ``--n-episodes``
    (contract flag name) and canonical artifact naming.

Example:
    From the RoboTwin repo root (inside the robotwin-headless container)::

        python scripts/run_eval.py \\
            --task_name beat_block_hammer --task_config demo_clean \\
            --policy_server_addr localhost:8000 \\
            --n-episodes 1 --log_dir ./logs --save_video

    Artefacts land in ``./logs/eval/beat_block_hammer--<ts>/``:
    ``eval.log`` (rolling success rate + FINAL RESULTS) and, if saved,
    ``episode_000_success={True|False}.mp4``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import runpy
import sys
from datetime import datetime


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_UPSTREAM_SCRIPT = os.path.join(_REPO_ROOT, "script", "run_eval_ws.py")
_UPSTREAM_UTILS_DIR = os.path.join(_REPO_ROOT, "script")


def _list_tasks() -> list:
    if _UPSTREAM_UTILS_DIR not in sys.path:
        sys.path.insert(0, _UPSTREAM_UTILS_DIR)
    from robotwin_run_utils import list_robotwin_task_names
    return list_robotwin_task_names()


def _parse_wrapper_args():
    tasks = _list_tasks()
    parser = argparse.ArgumentParser(
        description="RoboTwin eval client (benchmark-env-generator contract wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000")
    parser.add_argument("--policy", type=str, default="websocketPolicy")
    parser.add_argument("--n-episodes", "--n_episodes", dest="n_episodes",
                        type=int, default=5,
                        help="Episodes to evaluate (contract flag; translated to upstream --num_trials).")
    parser.add_argument("--instruction_type", type=str, default="unseen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_type", type=str, default="qpos",
                        choices=("qpos", "ee", "delta_ee"))
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Base log dir; eval artefacts land under <log_dir>/eval/<task>--<ts>/.")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")
    parser.add_argument("--list_tasks", action="store_true")
    args = parser.parse_args()
    if not args.list_tasks:
        if not args.task_name:
            parser.error("--task_name is required unless --list_tasks")
        if args.task_name not in tasks:
            parser.error(f"Unknown task_name {args.task_name!r}. Use --list_tasks.")
    return args


_EPISODE_MP4_RE = re.compile(r"^episode=(?P<idx>\d+)--success=(?P<succ>True|False)--task=.*\.mp4$")


def _canonicalize_eval_artifacts(run_dir: str) -> None:
    """Rename per-episode mp4s to the contract's episode_{idx:03d}_success={bool}.mp4 format."""
    for path in sorted(glob.glob(os.path.join(run_dir, "episode=*.mp4"))):
        m = _EPISODE_MP4_RE.match(os.path.basename(path))
        if not m:
            continue
        idx = int(m.group("idx"))
        succ = m.group("succ")
        target = os.path.join(run_dir, f"episode_{idx:03d}_success={succ}.mp4")
        if os.path.exists(target):
            os.remove(target)
        os.rename(path, target)


def main() -> int:
    args = _parse_wrapper_args()

    if args.list_tasks:
        sys.argv = [_UPSTREAM_SCRIPT, "--list_tasks"]
        runpy.run_path(_UPSTREAM_SCRIPT, run_name="__main__")
        return 0

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_dir = os.path.join(args.log_dir, "eval")
    run_dir = os.path.join(eval_log_dir, f"{args.task_name}--{date_str}")

    upstream_argv = [
        _UPSTREAM_SCRIPT,
        "--task_name", args.task_name,
        "--task_config", args.task_config,
        "--policy_server_addr", args.policy_server_addr,
        "--policy", args.policy,
        "--num_trials", str(args.n_episodes),
        "--instruction_type", args.instruction_type,
        "--seed", str(args.seed),
        "--action_type", args.action_type,
        "--log_dir", eval_log_dir,
    ]
    if args.save_video:
        upstream_argv.append("--save_video")
    else:
        upstream_argv.append("--no_save_video")
    sys.argv = upstream_argv

    print(f"# RoboTwin eval wrapper -> {upstream_argv}")
    print(f"# expected run_dir prefix: {eval_log_dir}/{args.task_name}--<ts>/")
    rc = 0
    try:
        runpy.run_path(_UPSTREAM_SCRIPT, run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)

    upstream_run_dirs = sorted(
        glob.glob(os.path.join(eval_log_dir, f"{args.task_name}--*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if upstream_run_dirs:
        actual_run_dir = upstream_run_dirs[0]
        try:
            _canonicalize_eval_artifacts(actual_run_dir)
        except Exception as e:
            print(f"# artefact rename failed: {type(e).__name__}: {e}")
        print(f"Run directory: {actual_run_dir}")
    else:
        print(f"Run directory: (not found under {eval_log_dir})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
