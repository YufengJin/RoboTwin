"""Map RoboTwin get_obs() output to keys expected by policy-websocket / OpenVLA-style servers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _rgb_to_uint8_hwc(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb)
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)


def robotwin_obs_to_policy_dict(now_obs: Dict[str, Any], task_description: str) -> Dict[str, Any]:
    """Build observation dict for remote policy (RoboCasa / ManiSkill compatible image keys).

    Uses RoboTwin camera names from envs: typically head_camera, left_camera, right_camera
    when rgb collection is enabled in task_config.

    Mapping:
      - robot0_agentview_left_image: head_camera if present, else left_camera
      - robot0_agentview_right_image: right_camera if present, else head or left fallback
      - robot0_eye_in_hand_image: left_camera (wrist) if present, else head
    """
    obs = now_obs.get("observation") or {}

    def _cam_rgb(name: str):
        if name not in obs or not isinstance(obs[name], dict):
            return None
        if "rgb" not in obs[name]:
            return None
        return _rgb_to_uint8_hwc(obs[name]["rgb"])

    head = _cam_rgb("head_camera")
    left = _cam_rgb("left_camera")
    right = _cam_rgb("right_camera")

    if head is not None and left is not None and right is not None:
        primary = head
        secondary = right
        wrist = left
    elif head is not None:
        primary = head
        secondary = right if right is not None else head
        wrist = left if left is not None else head
    elif left is not None:
        primary = left
        secondary = right if right is not None else left
        wrist = left
    else:
        raise ValueError(
            "No RGB in observation: enable data_type.rgb and camera collection in task_config "
            "(expected head_camera and/or wrist cameras)."
        )

    out: Dict[str, Any] = {
        "robot0_agentview_left_image": primary,
        "robot0_agentview_right_image": secondary,
        "robot0_eye_in_hand_image": wrist,
        "task_description": task_description,
    }
    return out
