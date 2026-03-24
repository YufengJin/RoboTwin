"""Shared helpers for run_demo_ws.py / run_eval_ws.py (WebSocket policy client)."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from robotwin_policy_obs import robotwin_obs_to_policy_dict

import numpy as np
import yaml


def get_repo_root() -> str:
    """RoboTwin repository root (parent of script/)."""
    return os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def get_task_config_dir() -> str:
    """Directory containing task_config/*.yml (trailing sep for CONFIGS_PATH + file joins)."""
    return os.path.join(get_repo_root(), "task_config") + os.sep


def list_robotwin_task_names() -> List[str]:
    """Task modules under envs/*.py (same convention as script/eval_policy.py class_decorator)."""
    envs_dir = os.path.join(get_repo_root(), "envs")
    skip = frozenset(
        {
            "__init__.py",
            "_base_task.py",
            "_GLOBAL_CONFIGS.py",
        }
    )
    names: List[str] = []
    for fname in os.listdir(envs_dir):
        if not fname.endswith(".py") or fname in skip or fname.startswith("_"):
            continue
        path = os.path.join(envs_dir, fname)
        if os.path.isfile(path):
            names.append(fname[:-3])
    return sorted(names)


def get_embodiment_config(robot_file: str) -> dict:
    path = robot_file
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(get_repo_root(), path.lstrip("./")))
    robot_config_file = os.path.join(path.rstrip("/"), "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_eval_args_from_yaml(task_name: str, task_config: str) -> dict:
    """Load task_config/{task_config}.yml and merge embodiment/camera fields (see eval_policy.main)."""
    repo = get_repo_root()
    cfg_path = os.path.join(repo, "task_config", f"{task_config}.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(get_task_config_dir(), "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type: str):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(os.path.join(get_task_config_dir(), "_camera_config.yml"), "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be length 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def policy_infer_init(
    TASK_ENV,
    policy,
    task_name: str,
    instruction: str,
) -> None:
    """Send one init message with action_dim for servers that expect it."""
    obs = TASK_ENV.get_obs()
    ja = obs.get("joint_action") or {}
    if "vector" in ja:
        dim = int(np.asarray(ja["vector"]).ravel().shape[0])
    else:
        dim = 0
    init_obs: Dict[str, Any] = {
        "action_dim": dim,
        "task_name": task_name,
        "task_description": instruction,
    }
    policy.infer(init_obs)


def execute_policy_action_chunk(
    TASK_ENV,
    policy,
    task_description: str,
    action_type: str,
) -> None:
    """One policy.infer; apply all returned action rows via take_action (chunked execution)."""
    pol_obs = robotwin_obs_to_policy_dict(TASK_ENV.get_obs(), task_description)
    res = policy.infer(pol_obs)
    actions = np.asarray(res["actions"], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    for row in actions:
        TASK_ENV.take_action(np.asarray(row, dtype=np.float64).ravel(), action_type=action_type)


def infer_action_dim_from_obs(now_obs: dict) -> int:
    ja = now_obs.get("joint_action") or {}
    if "vector" in ja:
        return int(np.asarray(ja["vector"]).ravel().shape[0])
    return 0


def websocket_policy_rollout(
    TASK_ENV,
    policy,
    task_name: str,
    instruction: str,
    action_type: str,
    *,
    save_frames: bool = False,
) -> Tuple[bool, list]:
    """Run policy until step_lim or eval_success. Optionally collect (p,s,w) uint8 frames after each chunk."""
    policy.reset()
    policy_infer_init(TASK_ENV, policy, task_name, instruction)
    frames: list = []
    succ = False
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        if save_frames:
            po = robotwin_obs_to_policy_dict(TASK_ENV.get_obs(), instruction)
            frames.append(
                (
                    np.array(po["robot0_agentview_left_image"], copy=True),
                    np.array(po["robot0_agentview_right_image"], copy=True),
                    np.array(po["robot0_eye_in_hand_image"], copy=True),
                )
            )
        execute_policy_action_chunk(TASK_ENV, policy, instruction, action_type)
        if TASK_ENV.eval_success:
            succ = True
            break
    return succ, frames


def _import_generate_episode_descriptions():
    du = os.path.join(get_repo_root(), "description", "utils")
    if du not in sys.path:
        sys.path.insert(0, du)
    from generate_episode_instructions import generate_episode_descriptions

    return generate_episode_descriptions


def run_robotwin_ws_episode(
    TASK_ENV,
    policy,
    task_name: str,
    args: dict,
    now_id: int,
    now_seed: int,
    instruction_type: str,
    sample_n: int,
    action_type: str,
    *,
    save_frames: bool = False,
    clear_cache_after: bool = False,
) -> Tuple[bool, list, int, bool, str, int]:
    """Expert validation + instruction + WebSocket policy rollout.

    Returns (policy_success, video_frames, next_now_seed, ran_rollout, instruction, take_action_cnt).
    take_action_cnt is 0 if ran_rollout is False.
    """
    from envs.utils.create_actor import UnStableError

    generate_episode_descriptions = _import_generate_episode_descriptions()

    render_freq = args["render_freq"]
    args["render_freq"] = 0
    episode_info = None
    try:
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info = TASK_ENV.play_once()
        TASK_ENV.close_env()
    except UnStableError:
        try:
            TASK_ENV.close_env()
        except Exception:
            pass
        args["render_freq"] = render_freq
        return False, [], now_seed + 1, False, "", 0
    except Exception:
        try:
            TASK_ENV.close_env()
        except Exception:
            pass
        args["render_freq"] = render_freq
        return False, [], now_seed + 1, False, "", 0

    args["render_freq"] = render_freq

    if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
        return False, [], now_seed + 1, False, "", 0

    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    episode_info_list = [episode_info["info"]]
    results = generate_episode_descriptions(task_name, episode_info_list, sample_n)
    instruction = str(np.random.choice(results[0][instruction_type]))
    TASK_ENV.set_instruction(instruction=instruction)

    succ, frames = websocket_policy_rollout(
        TASK_ENV,
        policy,
        task_name,
        instruction,
        action_type,
        save_frames=save_frames,
    )

    steps = int(getattr(TASK_ENV, "take_action_cnt", 0))
    TASK_ENV.close_env(clear_cache=clear_cache_after)
    if getattr(TASK_ENV, "render_freq", 0):
        try:
            TASK_ENV.viewer.close()
        except Exception:
            pass

    return succ, frames, now_seed + 1, True, instruction, steps
