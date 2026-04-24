#!/usr/bin/env python3
"""Random-action WebSocket policy server for RoboTwin smoke tests.

Purpose:
    Serves a random-sampling BasePolicy over the ``policy_websocket`` WebSocket
    protocol so ``scripts/run_demo.py`` and ``scripts/run_eval.py`` can be
    exercised end-to-end without a trained policy or a dataset. Intentionally
    minimal: no env construction, no SAPIEN / torch / gymnasium imports, no GUI,
    no disk I/O beyond stderr logging. The handshake metadata advertises
    ``action_dim`` (default 14, matching the dual-arm Piper qpos layout used by
    all 50 task modules in ``envs/``) so clients with strict-dim guards accept
    the connection.

Example:
    Run the server (container or host)::

        python tests/test_random_policy_server.py --host 0.0.0.0 --port 8000 --action_dim 14

    Then from another shell::

        python scripts/run_demo.py --task_name beat_block_hammer \\
            --task_config demo_clean --policy_server_addr localhost:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

import numpy as np

from policy_websocket import BasePolicy, WebsocketPolicyServer


class RandomPolicy(BasePolicy):
    """Returns a (chunk_len, action_dim) tensor of small uniform-random actions each infer()."""

    def __init__(self, action_dim: int, chunk_len: int, low: float, high: float, seed: int) -> None:
        self._action_dim = action_dim
        self._chunk_len = chunk_len
        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        actions = self._rng.uniform(
            self._low, self._high, size=(self._chunk_len, self._action_dim)
        ).astype(np.float32)
        return {"actions": actions}

    def reset(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action_dim", type=int, default=14,
                        help="Dual-arm Piper qpos layout is 14 (6 DOF + 1 gripper per arm).")
    parser.add_argument("--chunk_len", type=int, default=16,
                        help="Rows returned per infer(); clients iterate these via take_action.")
    parser.add_argument("--low", type=float, default=-0.05)
    parser.add_argument("--high", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    policy = RandomPolicy(
        action_dim=args.action_dim,
        chunk_len=args.chunk_len,
        low=args.low,
        high=args.high,
        seed=args.seed,
    )
    metadata = {
        "action_dim": args.action_dim,
        "chunk_len": args.chunk_len,
        "policy_name": "random",
        "version": "smoke-test-1",
    }
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    logging.info("starting random policy server on ws://%s:%d (action_dim=%d, chunk_len=%d)",
                 args.host, args.port, args.action_dim, args.chunk_len)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
