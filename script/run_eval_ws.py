#!/usr/bin/env python3
"""
RoboTwin evaluation client (WebSocket / policy-websocket).

与 script/eval_policy.py 的区别：策略推理走 WebsocketClientPolicy；日志与视频写入
<log_dir>/<task_name>--<YYYYMMDD_HHMMSS>/ 下的 eval.log。

动作空间：双臂 qpos/ee 等，维度由 embodiment 决定；与 OpenVLA 默认 7D 单臂不直接兼容。

可运行任务数：50（见 python script/run_eval_ws.py --list_tasks）。

Usage (from RoboTwin repo root):
    python script/run_eval_ws.py --task_name beat_block_hammer --task_config demo_clean \\
        --policy_server_addr localhost:8000
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_REPO_ROOT)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "policy"), os.path.join(_REPO_ROOT, "description", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import signal
from datetime import datetime

import numpy as np


def _websocket_policy_class():
    try:
        from policy_websocket import WebsocketClientPolicy
    except ImportError as e:
        raise ImportError(
            "policy-websocket is required. pip install 'policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git'"
        ) from e
    return WebsocketClientPolicy


def log(msg: str, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def save_rollout_video(frames, episode_idx, success, instruction, output_dir):
    import imageio

    if not frames:
        return None
    os.makedirs(output_dir, exist_ok=True)
    tag = (
        instruction.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:40]
    )
    filename = f"episode={episode_idx}--success={success}--task={tag}.mp4"
    mp4_path = os.path.join(output_dir, filename)
    writer = imageio.get_writer(mp4_path, fps=10, format="FFMPEG", codec="libx264")
    for p, s, w in frames:
        frame = np.concatenate([p, s, w], axis=1)
        writer.append_data(frame)
    writer.close()
    log(f"Saved rollout video: {mp4_path}")
    return mp4_path


def parse_args():
    from robotwin_run_utils import list_robotwin_task_names

    tasks = list_robotwin_task_names()
    parser = argparse.ArgumentParser(
        description="RoboTwin WebSocket evaluation client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="envs.<task_name> task (required unless --list_tasks)",
    )
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000")
    parser.add_argument("--policy", type=str, default="websocketPolicy")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--instruction_type", type=str, default="unseen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_type", type=str, default="qpos", choices=("qpos", "ee", "delta_ee"))
    parser.add_argument("--log_dir", type=str, default="./eval_logs_ws")
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")
    parser.add_argument("--list_tasks", action="store_true")
    args = parser.parse_args()
    if not args.list_tasks:
        if not args.task_name:
            parser.error("--task_name is required unless --list_tasks")
        if args.task_name not in tasks:
            parser.error(f"Unknown task_name {args.task_name!r}. Use --list_tasks.")
    return args


def main():
    args = parse_args()

    from robotwin_run_utils import (
        build_eval_args_from_yaml,
        class_decorator,
        list_robotwin_task_names,
        run_robotwin_ws_episode,
    )

    if args.list_tasks:
        tasks = list_robotwin_task_names()
        print(f"RoboTwin runnable tasks: {len(tasks)}")
        for t in tasks:
            print(f"  {t}")
        return

    yargs = build_eval_args_from_yaml(args.task_name, args.task_config)
    yargs["eval_video_log"] = False
    yargs["eval_video_save_dir"] = None
    yargs["eval_mode"] = True

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, f"{args.task_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "eval.log")
    log_file = open(log_path, "w")

    addr = args.policy_server_addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 8000

    log("=" * 60, log_file)
    log("RoboTwin WebSocket Eval Run", log_file)
    log("=" * 60, log_file)
    log(f"  task_name:        {args.task_name}", log_file)
    log(f"  task_config:      {args.task_config}", log_file)
    log(f"  num_trials:       {args.num_trials}", log_file)
    log(f"  policy_server:    ws://{host}:{port}", log_file)
    log(f"  log_dir:          {run_dir}", log_file)
    log(f"  action_type:      {args.action_type}", log_file)
    log("=" * 60, log_file)

    WebsocketClientPolicy = _websocket_policy_class()
    policy = WebsocketClientPolicy(host=host, port=port)
    meta = policy.get_server_metadata()
    log(f"Server metadata: {meta}", log_file)

    TASK_ENV = class_decorator(args.task_name)

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        try:
            policy.close()
        except Exception:
            pass
        if not log_file.closed:
            log_file.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    now_id = 0
    now_seed = 100000 * (1 + args.seed)
    sample_n = max(100, args.num_trials)
    successes = []
    lengths = []
    expert_fail_streak = 0
    max_expert_fails = max(200, args.num_trials * 50)
    ep_idx = 0

    try:
        while ep_idx < args.num_trials:
            if expert_fail_streak > max_expert_fails:
                log("Too many expert/scene failures; stopping early.", log_file)
                break
            clear_cache = (ep_idx + 1) % yargs.get("clear_cache_freq", 5) == 0
            log(f"\n--- Episode {ep_idx + 1}/{args.num_trials} ---", log_file)
            succ, frames, now_seed, ran, instruction, n_steps = run_robotwin_ws_episode(
                TASK_ENV,
                policy,
                args.task_name,
                yargs,
                now_id,
                now_seed,
                args.instruction_type,
                sample_n,
                args.action_type,
                save_frames=args.save_video,
                clear_cache_after=clear_cache,
            )
            if not ran:
                expert_fail_streak += 1
                log("  Expert/scene validation failed; retrying with new seed.", log_file)
                continue
            expert_fail_streak = 0
            successes.append(succ)
            lengths.append(n_steps)
            instr_short = (instruction or "")[:80]
            log(f"  Episode {ep_idx}: {'SUCCESS' if succ else 'FAILURE'} | {instr_short}...", log_file)
            if args.save_video and frames:
                save_rollout_video(frames, ep_idx, succ, instruction or args.task_name, run_dir)
            sr = sum(successes) / len(successes) * 100
            log(f"Running success rate: {sum(successes)}/{len(successes)} ({sr:.1f}%)", log_file)
            ep_idx += 1
            now_id += 1

        if successes:
            success_rate = float(np.mean(successes))
            avg_len = float(np.mean(lengths)) if lengths else 0.0
            log("\n" + "=" * 60, log_file)
            log("FINAL RESULTS", log_file)
            log("=" * 60, log_file)
            log(f"Policy:           {args.policy}", log_file)
            log(f"Task:             {args.task_name}", log_file)
            log(f"Success rate:     {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
            log(f"Avg ep length:    {avg_len:.1f}", log_file)
            log(f"Total episodes:   {len(successes)}", log_file)
            log(f"Total successes:  {sum(successes)}", log_file)
            log("=" * 60, log_file)

        log(f"\nLog saved to: {log_path}", log_file)
        print(f"\nLog saved to: {log_path}")
        print(f"Run directory: {run_dir}")
    finally:
        policy.close()
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
