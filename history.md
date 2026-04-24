# RoboTwin skill-run history

One-shot log of the `benchmark-env-generator` skill run on this repo. Regenerated (overwritten) every time the skill runs. For the reproducibility recipe see [`install.md`](install.md); for benchmark semantics see [`benchmark.md`](benchmark.md).

## Run metadata

| Key | Value |
|-----|-------|
| Skill | `superpowers-plugin / benchmark-env-generator` |
| Repo | `/home/yjin/repos/RoboTwin` @ `main` (commit `824e1ec`) |
| Classified as | **IL** (imitation learning) |
| Run date | 2026-04-24 |
| Host OS | Linux 5.15.0-139-generic |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| NVIDIA driver | 570.211.01 |
| Docker | v29.4.0, compose v5.1.2 |

## Classification evidence

| Criterion | Result | Evidence |
|-----------|--------|----------|
| Main: env defines reward signal | **No** | `envs/_base_task.py` exposes `take_action()` (no return) + `check_success()` (bool) only; no `def step(self, ...)`, no `compute_reward`, no `reward_fn=`. Evaluated across 50 `envs/*.py` modules. |
| Aux 1: `gym.register(` | 0 | Not present. |
| Aux 2: VecEnv / AsyncVectorEnv / SyncVectorEnv | 0 | Not present. |
| Aux 3: horizon / max_episode_steps on envs | 0 | Per-task step limit in `task_config/_eval_step_limit.yml` only (not a gym horizon). |
| Aux 4: JAX stateless env patterns | 0 | `jax` imports exist under `policy/pi0/` (model-side); env is PyTorch + SAPIEN. |
| Aux 5: `observation_space` / `action_space` declared | 0 | `Base_Task` inherits `gym.Env` but does not declare spaces. |
| Aux 6: in-tree `algorithms/` or `examples/training_examples/` | 0 | Not present. |
| Aux 7: README keywords (PPO, SAC, policy gradient, reward shaping) | 0 | No matches in `README.md`. |
| Aux 8: shaped-reward YAMLs | 0 | `task_config/*.yml` defines camera/embodiment/step-limit only. |
| **Aux score** | **0 / 8** | Decision: **IL**. |

Action dim probe: 14 (6-DOF arm + 1 gripper) × 2 arms = 14 per step (`qpos` action_type). Source: runtime `obs["joint_action"]["vector"]` shape in `script/robotwin_run_utils.py:147`. Confirmed at L1 smoke: `L1 OK dim= 14`.

## Generated / modified files

| Kind | Path | Purpose |
|------|------|---------|
| NEW | `tests/test_random_policy_server.py` | Pure-WebSocket random policy server (no env imports); advertises `action_dim=14` in handshake metadata. Mandatory per skill contract. |
| NEW | `scripts/__init__.py` | Empty; prevents namespace clash with `script/`. |
| NEW | `scripts/run_demo.py` | WebSocket-client wrapper around `script/run_demo_ws.py` with contract flags (`--log_dir` default `./logs`, `demo.log` tee, `demo.mp4` rename, Purpose/Example docstring). |
| NEW | `scripts/run_eval.py` | WebSocket-client wrapper around `script/run_eval_ws.py`; translates `--n-episodes` → `--num_trials`, renames per-episode MP4s to `episode_{ep:03d}_success={bool}.mp4`. |
| NEW | `docker/nvidia_icd.json` | Vulkan ICD manifest (NVIDIA canonical, api_version 1.3.194). COPYed into `/usr/share/vulkan/icd.d/`. |
| NEW | `docker/nvidia_layers.json` | VK_LAYER_NV_optimus manifest. COPYed into `/usr/share/vulkan/implicit_layer.d/`. |
| NEW | `install.md` | Reproducibility recipe (English). |
| NEW | `history.md` | This file. |
| NEW | `benchmark.md` | Benchmark description + 50-task inventory (English). |
| EDIT | `docker/Dockerfile` | Added Vulkan-ICD COPY + `ENV VK_ICD_FILENAMES`; `RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root`. |
| EDIT | `docker/entrypoint.sh` | Added `rm -f /tmp/entrypoint_done` at start and `touch /tmp/entrypoint_done` before `exec "$@"`. |
| EDIT | `docker/docker-compose.headless.yaml` | Added `- XDG_RUNTIME_DIR=/tmp/runtime-root` to `environment`. |
| EDIT | `docker/docker-compose.x11.yaml` | Same as headless. |
| EDIT | `docker/README.md` | Rewritten in English; same structure as before. |

Intentionally unchanged: `pyproject.toml`, `uv.lock`, `script/run_demo_ws.py`, `script/run_eval_ws.py`, `script/robotwin_run_utils.py`, `script/_download_assets.sh`, all 50 `envs/*.py`.

## Capability probes inside the container

```text
$ docker exec robotwin-headless python -c "import torch; print('cuda:', torch.cuda.is_available()); import sapien; print('sapien:', sapien.__version__)"
cuda: True
sapien: 3.0.0b1

$ docker exec robotwin-headless ls /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/implicit_layer.d/nvidia_layers.json /tmp/runtime-root
/usr/share/vulkan/icd.d/nvidia_icd.json
/usr/share/vulkan/implicit_layer.d/nvidia_layers.json
/tmp/runtime-root/   (empty dir, mode 700)

$ docker exec robotwin-headless ls /tmp/entrypoint_done
-rw-r--r-- 1 root root 0 Apr 24 13:25 /tmp/entrypoint_done
```

Vulkan ICD + runtime dir + sentinel file all in place.

## Four-tier smoke test results

Container: `robotwin-headless` (built from `docker/docker-compose.headless.yaml`, image `robotwin:latest` digest `sha256:dbef4c0e779c...`). Smoke task: `beat_block_hammer` + `demo_clean` (step limit 400).

### L1 — reset + 10 steps (no reward assertion) → **PASS**

Command:

```bash
docker exec robotwin-headless bash -lc 'cd /workspace/RoboTwin && python -c "
import sys, numpy as np; sys.path.insert(0, \"script\")
from robotwin_run_utils import class_decorator, build_eval_args_from_yaml
env = class_decorator(\"beat_block_hammer\")
y = build_eval_args_from_yaml(\"beat_block_hammer\", \"demo_clean\"); y[\"eval_mode\"]=True; y[\"render_freq\"]=0
env.setup_demo(now_ep_num=0, seed=100000, is_test=True, **y)
dim = int(np.asarray(env.get_obs()[\"joint_action\"][\"vector\"]).ravel().shape[0])
for _ in range(10): env.take_action(np.random.uniform(-0.05,0.05,dim), action_type=\"qpos\")
env.close_env(); print(\"L1 OK dim=\", dim)"'
```

Last stdout lines:

```
step: 1 / 400 ... step: 10 / 400
L1 OK dim= 14
```

### L2 — reset + 10 steps asserting finite reward → **FAIL (expected)**

Command (degraded to a compatibility probe, since upstream does not expose `step(action) -> (..., reward, ...)`):

```bash
docker exec robotwin-headless bash -lc 'cd /workspace/RoboTwin && python -c "
from envs._base_task import Base_Task
print(\"L2 SKIP: RoboTwin envs expose take_action(), not gym-style step(). Evaluation is via check_success(); see history.md note.\")
exit(2)"' || echo "L2 recorded as expected-fail"
```

Stdout:

```
L2 SKIP: RoboTwin envs expose take_action(), not gym-style step(). Evaluation is via check_success(); see history.md note.
```

**Why FAIL is the right outcome, not a skill bug.** The skill's L2 contract checks for a finite `reward` from `env.step()`. RoboTwin is a pure IL benchmark: its `Base_Task` never returns a reward — success is decided by `TASK_ENV.check_success() -> bool` at the end of an episode. Synthesising a dummy reward to make L2 green would mask this semantic mismatch and is explicitly rejected by the skill's "principled FAIL" policy. The smoke harness proceeds with L3_IL, which is the correct evaluation path for this repo.

### L3_IL — random policy server + run_demo + run_eval → **PASS**

Step 1 — start server (detached):

```bash
docker exec -d robotwin-headless bash -lc \
  'cd /workspace/RoboTwin && python tests/test_random_policy_server.py --host 0.0.0.0 --port 8000 --action_dim 14 --chunk_len 16 > /tmp/random_policy_server.log 2>&1'
```

Server log (first 3 lines):

```
2026-04-24 13:26:53,420 INFO root: starting random policy server on ws://0.0.0.0:8000 (action_dim=14, chunk_len=16)
2026-04-24 13:26:53,421 INFO websockets.server: server listening on 0.0.0.0:8000
2026-04-24 13:26:53,421 INFO policy_websocket.websocket_server: PolicyServer listening on ws://0.0.0.0:8000
```

Port poll: `server up after 1s`.

Step 2 — `scripts/run_demo.py` (1 reset, save_video):

```bash
docker exec robotwin-headless bash -lc \
  'cd /workspace/RoboTwin && python scripts/run_demo.py \
     --task_name beat_block_hammer --task_config demo_clean \
     --policy_server_addr localhost:8000 \
     --num_resets 1 --log_dir ./logs --save_video'
```

Last 5 stdout lines (excluding per-step progress):

```
Episode 0: FAILURE
Saved rollout video: ./logs/demo/beat_block_hammer--20260424_132713/episode=0--success=False--task=grab_the_hammer_with_claw_and_smooth_hea.mp4
Done.
Run directory: ./logs/demo/beat_block_hammer--20260424_132713
Log saved to:  ./logs/demo/beat_block_hammer--20260424_132713/demo.log
```

Artefacts (post-canonicalize):

```
./logs/demo/beat_block_hammer--20260424_132713/
  demo.log  (12068 bytes; tee of stdout, head includes wrapper argv)
  demo.mp4  (34268 bytes; side-by-side 3-cam stitch, 10 fps, H.264)
```

Server handshake seen by client (proves WebSocket round-trip works):

```
Server metadata: {'action_dim': 14, 'chunk_len': 16, 'policy_name': 'random', 'version': 'smoke-test-1'}
```

Step 3 — `scripts/run_eval.py` (n-episodes=1, save_video):

```bash
docker exec robotwin-headless bash -lc \
  'cd /workspace/RoboTwin && python scripts/run_eval.py \
     --task_name beat_block_hammer --task_config demo_clean \
     --policy_server_addr localhost:8000 \
     --n-episodes 1 --log_dir ./logs --save_video'
```

Tail of `eval.log` (`FINAL RESULTS` block):

```
Running success rate: 0/1 (0.0%)

============================================================
FINAL RESULTS
============================================================
Policy:           websocketPolicy
Task:             beat_block_hammer
Success rate:     0.0000 (0%)
Avg ep length:    400.0
Total episodes:   1
Total successes:  0
============================================================
```

Artefacts:

```
./logs/eval/beat_block_hammer--20260424_132837/
  eval.log                             (1112 bytes; rolling rate + FINAL RESULTS)
  episode_000_success=False.mp4        (35305 bytes; contract-format filename)
```

Step 4 — teardown:

```bash
docker exec robotwin-headless bash -lc 'pkill -f test_random_policy_server.py || true'
# → "no server process" after ~1 s
```

## Final report

| Tier | Result | Notes |
|------|--------|-------|
| L1 reset+step | **PASS** | dim=14 confirmed, 10 steps completed, close_env clean. |
| L2 reward-finite | **FAIL (expected)** | IL benchmark has no `step()` reward; success is `check_success()`. Documented mismatch; not a skill-execution bug. |
| L3_IL random policy server | **PASS** | WebSocket listener up on `:8000`, handshake metadata round-trips. |
| L3_IL run_demo.py | **PASS** | 400 steps, WebSocket chunked infer loop, demo.log + demo.mp4 produced with contract names. |
| L3_IL run_eval.py | **PASS** | 1 episode, rolling success rate + `FINAL RESULTS` block, `episode_000_success=False.mp4` produced. |

The image is **end-to-end runnable** on first `docker compose up` without any additional checkpoint, dataset, or policy install. A real policy can be dropped in by replacing the random server with any `policy_websocket`-compatible server and pointing the clients at its `host:port`.

## Open follow-ups

- `pkill` permission error inside the container when the random server runs under `pid: host` — harmless (the process did exit, just via `docker exec` receiving SIGTERM instead of the target pid). Not blocking.
- Upstream warnings visible in every run (`pkg_resources deprecated`, `missing pytorch3d`, `Warp DeprecationWarning: warp.torch.device_from_torch`) — not actionable by this skill; noisy but non-fatal.
- `action_dim=14` is hard-coded in the random server handshake metadata. Multi-embodiment configs with a different qpos width must pass `--action_dim N` or the handshake will advertise the wrong value. Documented in `benchmark.md`.
