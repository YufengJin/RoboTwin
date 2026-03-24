#!/usr/bin/env python3
"""
RoboTwin demo client (WebSocket / policy-websocket).

与 script/eval_policy.py 的区别：策略推理走 WebsocketClientPolicy（与 LIBERO/RoboCasa 同协议），
而不是 TCP ModelClient + policy/*/deploy_policy.py。

动作空间：RoboTwin 的 take_action 默认期望双臂关节 + 双夹爪（action_type=qpos 等），
维度由 embodiment 决定。OpenVLA 等 7D 单臂策略不能直接使用，除非 Policy Server 侧做适配或专门训练。

可运行任务数：envs/ 下任务模块共 50 个（与 class_decorator 约定一致）。
列出任务：python script/run_demo_ws.py --list_tasks

Usage (from RoboTwin repo root):
    python script/run_demo_ws.py --task_name beat_block_hammer --task_config demo_clean \\
        --policy_server_addr localhost:8000
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_REPO_ROOT)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "policy"), os.path.join(_REPO_ROOT, "description", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atexit
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
    print(f"Saved rollout video: {mp4_path}")
    return mp4_path


def parse_args():
    from robotwin_run_utils import list_robotwin_task_names

    tasks = list_robotwin_task_names()
    parser = argparse.ArgumentParser(
        description="RoboTwin WebSocket demo (no eval log)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="envs.<task_name> task (required unless --list_tasks)",
    )
    parser.add_argument("--task_config", type=str, default="demo_clean", help="task_config/*.yml basename")
    parser.add_argument("--policy_server_addr", type=str, default="localhost:8000")
    parser.add_argument("--policy", type=str, default="websocketPolicy", help="Label for logging only")
    parser.add_argument("--num_resets", type=int, default=3, help="Successful policy episodes to run")
    parser.add_argument("--instruction_type", type=str, default="unseen", help="Key for generate_episode_descriptions")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (matches eval_policy st_seed scale)")
    parser.add_argument("--action_type", type=str, default="qpos", choices=("qpos", "ee", "delta_ee"))
    parser.add_argument("--demo_log_dir", type=str, default="./demo_log_ws")
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="Print RoboTwin task count and names, then exit",
    )
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
    run_dir = os.path.join(args.demo_log_dir, f"{args.task_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)

    addr = args.policy_server_addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 8000

    print("=" * 60)
    print("RoboTwin Demo (WebSocket, no eval)")
    print("=" * 60)
    print(f"  task_name:        {args.task_name}")
    print(f"  task_config:      {args.task_config}")
    print(f"  num_resets:       {args.num_resets}")
    print(f"  policy_server:    ws://{host}:{port}")
    print(f"  demo_log_dir:     {run_dir}")
    print(f"  action_type:      {args.action_type}")
    print("=" * 60)

    WebsocketClientPolicy = _websocket_policy_class()
    policy = WebsocketClientPolicy(host=host, port=port)
    meta = policy.get_server_metadata()
    print(f"Server metadata: {meta}")

    TASK_ENV = class_decorator(args.task_name)

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        try:
            policy.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    from robotwin_run_utils import run_robotwin_ws_episode

    now_id = 0
    now_seed = 100000 * (1 + args.seed)
    sample_n = max(100, args.num_resets)
    completed = 0
    expert_fail_streak = 0
    max_expert_fails = max(200, args.num_resets * 50)

    try:
        while completed < args.num_resets:
            if expert_fail_streak > max_expert_fails:
                print("Too many expert/scene failures; stopping.")
                break
            clear_cache = (completed + 1) % yargs.get("clear_cache_freq", 5) == 0
            succ, frames, now_seed, ran, instruction, _steps = run_robotwin_ws_episode(
                TASK_ENV,
                policy,
                args.task_name,
                yargs,
                now_id,
                now_seed,
                args.instruction_type,
                sample_n,
                args.action_type,
                save_frames=True,
                clear_cache_after=clear_cache,
            )
            if not ran:
                expert_fail_streak += 1
                continue
            expert_fail_streak = 0
            print(f"  Episode {completed}: {'SUCCESS' if succ else 'FAILURE'}")
            if frames:
                save_rollout_video(frames, completed, succ, instruction or args.task_name, run_dir)
            completed += 1
            now_id += 1
    finally:
        policy.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
