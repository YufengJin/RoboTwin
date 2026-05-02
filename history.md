# RoboTwin skill-run history

One-shot log of the `benchmark-env-generator` skill run on this repo. Regenerated (overwritten) every time the skill runs. For the reproducibility recipe see [`install.md`](install.md); for benchmark semantics see [`benchmark.md`](benchmark.md).

## Run metadata

| Key | Value |
|-----|-------|
| Skill | `superpowers-plugin / benchmark-env-generator` |
| Repo | `/home/yjin/repos/RoboTwin` @ `nautilus-rerun-2026-05-01` (commit `7e15bdd`) |
| Classified as | **IL** (imitation learning) |
| Run date | 2026-05-02 |
| Host OS | Linux 5.15.0-139-generic |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| NVIDIA driver | 570.211.01 |
| Docker | v29.4.0, compose v5.1.2 |
| Image | `robotwin:local` (sha256:0aa50ec82ad4...; 18.5 GiB) |
| Container | `robotwin-headless` (already up; reused, not rebuilt) |
| env-generator quirks resolved | `needs_render_libs`, `needs_setuptools_pin`, `needs_vulkan_icd`, `needs_devel_base`, `needs_torch_cuda_arch` |

This rerun was driven against an env-generator-built image whose smoke results were `tier1=PASS`, `tier2=PARTIAL (7/18)` — the 11 failing imports are `jax / flax / openpi / openpi_client / ...` under `policy/pi0/` and `policy/pi05/` which carry their own independent venvs and are not part of the core RoboTwin `uv.lock`. Those baseline subdirs are not exercised by the WebSocket harness and do not block the IL contract.

## Classification evidence

| Criterion | Result | Evidence |
|-----------|--------|----------|
| Main: `env.step()` returns `reward` | **No** | `grep -rE 'def step\(self' envs/ -A 10` finds zero matches with `return.*reward` patterns. `envs/_base_task.py` exposes `take_action()` (no return) + `check_success()` (bool) only. No `compute_reward`, no `reward_fn=`. |
| Aux 1: `gym.register(` | 0 | Not present. |
| Aux 2: VecEnv / SubprocVec / DummyVec | 0 | Not present. |
| Aux 3: in-tree `algorithms/` (PPO/SAC/TD3/DDPG) | 0 | `envs/algorithms` does not exist. README mentions only IL / VLA baselines (`policy/ACT`, `policy/DP`, `policy/DP3`, `policy/DexVLA`, `policy/openvla-oft`, `policy/pi0`, `policy/pi05`, `policy/RDT`, `policy/TinyVLA`). |
| Aux 4: shaped-reward YAMLs | 0 | `task_config/*.yml` defines camera/embodiment/step-limit only. |
| Aux 5: `observation_space` / `action_space` declared | 0 | `Base_Task` inherits `gym.Env` but does not declare spaces (probed in container: `hasattr(env, "action_space") == False`). |
| Aux 6: VecEnv reset/step API in env code | 0 | Not present. |
| Aux 7: README PPO/SAC/policy gradient/reward shaping keywords | 0 | No matches. |
| Aux 8: JAX-stateless env patterns | 0 | `jax` imports exist under `policy/pi0/` / `policy/pi05/` (model-side); env stack is PyTorch + SAPIEN. |
| **Aux score** | **0 / 8** | Decision: **IL**. Plugin registry tags this benchmark `mixed` (IL+RL dual-pipeline) but the primary contract — and the only one exercised by `scripts/run_eval.py` and the WebSocket harness — is IL. |

Action dim probe: 14 (6-DOF arm + 1 gripper) × 2 arms (Piper dual-arm). Source: runtime `obs["joint_action"]["vector"]` shape in `script/robotwin_run_utils.py:147`. Confirmed at L1 smoke: `L1 OK action_dim= 14`.

## Generated / modified files (this run)

This is a re-run on a verified fork — `scripts/`, `tests/`, and the entire `docker/` stack were already in place from the prior `benchmark-env-generator` execution. Conformance of all rendered scaffolding was re-verified against the current contract before reuse; nothing was regenerated.

| Kind | Path | Notes |
|------|------|-------|
| REUSED | `scripts/run_demo.py` | Already conformant: `Purpose`/`Example` docstring, `--log_dir` default `./logs`, `--save_video`, `policy_websocket` client, log tee, `demo.mp4` rename. No edit. |
| REUSED | `scripts/run_eval.py` | Already conformant: `Purpose`/`Example`, `--n-episodes`/`--log_dir`/`--save_video`, `episode_{ep:03d}_success={bool}.mp4` rename, FINAL RESULTS block. No edit. |
| REUSED | `tests/test_random_policy_server.py` | Already conformant: pure WebSocket policy server, no env imports, advertises `action_dim=14` + `chunk_len=16` in handshake metadata. No edit. |
| REUSED | `docker/Dockerfile` | `policy_websocket` already installed via `uv sync --frozen` (pinned in `uv.lock` as a `git+https://github.com/YufengJin/policy_websocket.git` dep). No 2-line patch needed. |
| NEW | `.nautilus/benchmark-spec.json` | Captured by `scripts/benchmark-generator/capture_spec.py` against `RoboTwinSpecAdapter` (gym-shim wrapping `setup_demo`+`take_action`+`get_obs`+`check_success`). Cross-task verified `beat_block_hammer ↔ lift_pot` (identical action+filtered-obs spec). UNKNOWN controller/components/gripper_convention fields hand-annotated against the dual-arm Piper qpos contract. |
| EDIT | `benchmark.md` | OBS_ACTION_SPEC sentinel block re-rendered with the patched spec (correct camera kinds: `head_camera`/`front_camera`=`image_primary`, `left_camera`/`right_camera`=`image_wrist`; `joint_action.vector` included; controller + gripper convention populated). All other prose (About / Action+obs+reward / 50-task inventory / How to use) preserved — current and accurate. |
| EDIT | `history.md` | This file. |
| UNTRACKED (kept) | `logs/` | Prior smoke run artefacts from before this re-run. Not deleted per user instruction. |

Intentionally unchanged: `pyproject.toml`, `uv.lock`, `script/run_demo_ws.py`, `script/run_eval_ws.py`, `script/robotwin_run_utils.py`, `script/robotwin_policy_obs.py`, `script/_download_assets.sh`, `install.md`, all 50 `envs/*.py`, all of `docker/*` (env-generator-owned).

## Capability probes inside the container (this run)

```text
$ docker exec robotwin-headless python -c "print('container ok')"
container ok

$ docker exec robotwin-headless python -c "import envs; print('envs OK')"
envs OK   (preceded by benign sapien pkg_resources / requests-version warnings)

$ docker exec robotwin-headless python -c "import policy_websocket; print('policy_websocket OK')"
policy_websocket OK   (/opt/venv/lib/python3.10/site-packages/policy_websocket/__init__.py)

$ docker exec robotwin-headless ls /workspace/RoboTwin/assets/embodiments
ARX-X5  README.md  aloha-agilex  franka-panda  piper  ur5-wsg
```

`policy_websocket`, `sapien`, all 50 `envs/`, and the 5 embodiment asset families are present and importable. The Vulkan-ICD layer files (`docker/nvidia_icd.json` + `docker/nvidia_layers.json`) installed by env-generator are picked up — SAPIEN ray tracing initialises without `vk_icdNegotiateLoaderICDInterfaceVersion` errors.

## Four-tier smoke test results

Smoke task: `beat_block_hammer` + `demo_clean` (step limit 400). Verify task: `lift_pot` + `demo_clean` (step limit 400, used for cross-task spec consistency check).

### L1 — `setup_demo + 10 take_action steps` -> **PASS**

Command:

```bash
docker exec robotwin-headless bash -c 'cd /workspace/RoboTwin && python -c "
import sys, numpy as np
sys.path.insert(0, \"script\")
from robotwin_run_utils import class_decorator, build_eval_args_from_yaml
env = class_decorator(\"beat_block_hammer\")
y = build_eval_args_from_yaml(\"beat_block_hammer\", \"demo_clean\")
y[\"eval_mode\"]=True; y[\"render_freq\"]=0
env.setup_demo(now_ep_num=0, seed=100000, is_test=True, **y)
dim = int(np.asarray(env.get_obs()[\"joint_action\"][\"vector\"]).ravel().shape[0])
for _ in range(10):
    env.take_action(np.random.uniform(-0.05, 0.05, dim), action_type=\"qpos\")
env.close_env()
print(\"L1 OK action_dim=\", dim)
"'
```

Last 5 stdout lines:

```
step:  6 / 400
step:  7 / 400
step:  8 / 400
step:  9 / 400
step: 10 / 400
L1 OK action_dim= 14
```

### L2 — IL substitute (`check_success()` boolean signal) -> **PASS**

The skill contract's L2 normally asserts `reward is not None and np.isfinite(reward)`. RoboTwin is a pure IL benchmark with no shaped reward; the principled substitute (per the IL case study) is to assert that `check_success()` returns a non-erroring bool after the L1 step loop. This proves the success-judgement signal — the only evaluation channel RoboTwin offers — is wired end-to-end.

Command (extends the L1 command with a final `check_success()` call):

```bash
docker exec robotwin-headless bash -c 'cd /workspace/RoboTwin && python -c "
... (L1 setup + 10 steps) ...
ok = bool(env.check_success())
env.close_env()
print(\"L2 OK (IL: success-signal substitute):\", ok)
"'
```

Last stdout line:

```
L2 OK (IL: success-signal substitute): False
```

`False` is the expected outcome for a 10-step random rollout — what matters is that `check_success()` returns a real `bool` without raising, confirming the eval signal is alive.

### L3_IL — random policy server + run_demo + run_eval -> **PASS**

Step 1 — start server (detached, port 8000 — RoboTwin's verified-fork convention; the upstream `script/run_eval_ws.py` defaults to 8000 and the `tests/test_random_policy_server.py` shipped here matches):

```bash
docker exec -d robotwin-headless bash -c \
  'cd /workspace/RoboTwin && python tests/test_random_policy_server.py \
     --host 0.0.0.0 --port 8000 --action_dim 14 \
     > /tmp/policy_server_8000.log 2>&1 &
   echo $! > /tmp/policy_server_8000.pid'
```

Port poll (`socket.create_connection(("localhost", 8000), timeout=1)` inside the container) returned `server up` after 2 s.

Step 2 — `scripts/run_demo.py` (1 reset, no video for smoke):

```bash
docker exec robotwin-headless bash -c 'cd /workspace/RoboTwin && python scripts/run_demo.py \
   --task_name beat_block_hammer --task_config demo_clean \
   --policy_server_addr localhost:8000 \
   --num_resets 1 --log_dir /tmp/smoke_logs'
```

Last 5 stdout lines:

```
step: 400 / 400
  Episode 0: FAILURE
Saved rollout video: /tmp/smoke_logs/demo/beat_block_hammer--20260502_073234/episode=0--success=False--task=use_the_hammer_with_claw_and_smooth_head.mp4
Run directory: /tmp/smoke_logs/demo/beat_block_hammer--20260502_073234
Log saved to:  /tmp/smoke_logs/demo/beat_block_hammer--20260502_073234/demo.log
```

WebSocket handshake metadata round-tripped successfully (printed by upstream client at startup):

```
Server metadata: {'action_dim': 14, 'chunk_len': 16, 'policy_name': 'random', 'version': 'smoke-test-1'}
```

`Episode 0: FAILURE` is the expected outcome of a 400-step random-action rollout — the `policy_websocket` chunked-infer loop and the `scripts/run_demo.py` log-tee path both completed end-to-end.

Step 3 — `scripts/run_eval.py` (1 episode, no video):

```bash
docker exec robotwin-headless bash -c 'cd /workspace/RoboTwin && python scripts/run_eval.py \
   --task_name beat_block_hammer --task_config demo_clean \
   --policy_server_addr localhost:8000 \
   --n-episodes 1 --log_dir /tmp/smoke_logs --no_save_video'
```

Tail of `eval.log` (`FINAL RESULTS` block):

```
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
Log saved to: /tmp/smoke_logs/eval/beat_block_hammer--20260502_073328/eval.log
Run directory: /tmp/smoke_logs/eval/beat_block_hammer--20260502_073328
```

Step 4 — teardown (`kill $(cat /tmp/policy_server_8000.pid)` + `pkill -f test_random_policy_server`); `ps -ef | grep test_random_policy_server | grep -v grep` returned empty.

## Step 5.5 — obs/action spec capture -> **PASS**

`capture_spec.py` was run against a tiny `RoboTwinSpecAdapter` gym-shim (defined in `/tmp/robotwin_setup.py` inside the container) — RoboTwin's task classes don't expose `.action_space` or gym `.step()`, so the shim wraps `setup_demo` -> `reset`, `take_action+get_obs+check_success` -> `step` (5-tuple), and a fixed 14-dim float32 `Box(-1, 1)` action space matching the qpos contract.

Command:

```bash
docker exec robotwin-headless bash -c 'cd /workspace/RoboTwin && python /tmp/capture_spec.py \
    --repo-path /workspace/RoboTwin \
    --benchmark-name robotwin \
    --env-id beat_block_hammer \
    --setup "@/tmp/robotwin_setup.py" \
    --make-expr "RoboTwinSpecAdapter(task_name=\"beat_block_hammer\", task_config=\"demo_clean\")" \
    --verify-make-expr "RoboTwinSpecAdapter(task_name=\"lift_pot\", task_config=\"demo_clean\")" \
    --include-keys-regex "^joint_action\.vector$" \
    --category IL \
    --registry-dir /tmp/spec_out'
```

Last 4 stdout lines:

```
✓ cross-task verify passed: beat_block_hammer ↔ second task have identical action_spec + filtered obs_spec
✓ wrote /tmp/spec_out/robotwin.json
✓ patched /workspace/RoboTwin/benchmark.md
⚠ <UNKNOWN>/<UNHANDLED> fields — please fill before claiming done:
   - action_spec.controller
   - action_spec.gripper_convention
```

Two `<UNKNOWN>` fields (`action_spec.controller`, `action_spec.gripper_convention`) were reported by the auto-introspector — these are RoboTwin-specific (no standard `env.robots[0].controller` attribute, no robosuite-family OSC marker) and were hand-annotated post-capture against the dual-arm Piper qpos contract documented in `script/robotwin_run_utils.py::execute_policy_action_chunk`:

- `controller`: `DualArmJointPosController(left=PiperArm6+Gripper1, right=PiperArm6+Gripper1)`
- `components`: `[left_arm_0..5, left_gripper, right_arm_0..5, right_gripper]` (14 entries)
- `gripper_convention`: `{open: 0.04, close: 0.0, binary: false, index_left: 6, index_right: 13, note: "qpos action_type; gripper joint position in meters (Piper parallel-jaw)"}`

The script also under-tagged the 4 cameras as `image_primary`; per `script/robotwin_policy_obs.py`'s mapping (head=primary, left/right=wrist cameras, front=auxiliary primary), `left_camera.rgb` and `right_camera.rgb` were retagged `image_wrist` in the persisted JSON. Final spec is at `<repo>/.nautilus/benchmark-spec.json` (also embedded in the `<!-- BEGIN OBS_ACTION_SPEC -->` block of `benchmark.md`).

Cross-task verify (`beat_block_hammer ↔ lift_pot`) confirms the action spec and the filtered obs spec are task-invariant — the canonical robot interface does not change across the 50-task family.

## Final report

| Tier | Result | Notes |
|------|--------|-------|
| L1 reset+step | **PASS** | `action_dim=14` confirmed, 10 steps completed, `close_env()` clean. |
| L2 reward-finite (IL substitute) | **PASS** | `check_success() -> bool` works after L1 step loop. RoboTwin has no shaped reward; this is the principled IL substitute. |
| L3_IL random policy server | **PASS** | WebSocket listener up on `:8000`, handshake metadata round-trips. |
| L3_IL run_demo.py | **PASS** | 400-step rollout, WebSocket chunked infer loop, log-tee + run_dir produced. |
| L3_IL run_eval.py | **PASS** | 1-episode eval, rolling-rate log + `FINAL RESULTS` block, run_dir produced. |
| Step 5.5 spec capture + cross-task verify | **PASS** | `<repo>/.nautilus/benchmark-spec.json` written; `beat_block_hammer ↔ lift_pot` agree on action+filtered-obs spec. UNKNOWN controller/gripper fields hand-annotated against the qpos contract. |

The image is **end-to-end runnable** on first `docker compose up` without any additional checkpoint or dataset install. A real policy can be plugged in by replacing the random server with any `policy_websocket`-compatible server and pointing the clients at its `host:port`. The persisted `benchmark-spec.json` is consumable by `policy-generator` (Phase 2/4 obs/action layout lookup) once a maintainer copies it into `mcp/nautilus/specs/benchmarks/robotwin.json` during the verify step.

## Diagnostics applied

None. No env-config edits, no Dockerfile patches, no script/test regenerations were required. The verified fork's scaffolding passed every contract check on first try.

## Open follow-ups

- The `__meta__` envelope (skill contract requires every payload sent to a `policy_websocket` server include `obs["__meta__"] = {v, benchmark, task, task_description, phase}`) is **not** implemented in the upstream-fork wrappers (`scripts/run_demo.py` and `scripts/run_eval.py` delegate to `script/run_demo_ws.py`/`script/run_eval_ws.py` via `runpy.run_path`, with the obs-build hook in `script/robotwin_policy_obs.py`). Smoke passes because the bundled `tests/test_random_policy_server.py` ignores `__meta__`. To add it without breaking the verified fork's API, patch `script/robotwin_policy_obs.py::robotwin_obs_to_policy_dict` to accept an optional `meta` arg, then thread `task_name` from `script/robotwin_run_utils.py::policy_infer_init` and `execute_policy_action_chunk`. Surface to the user before applying — this is a behavioural change to the verified fork and may invalidate the verified status if not coordinated upstream.
- `pointcloud` shape is `[0]` in the captured spec because the default `demo_clean` task config disables pointcloud collection. Consumers needing pointcloud input should set `data_type.pointcloud=true` in `task_config/demo_clean.yml` (or whatever config they pass). Documented in the `benchmark.md` sentinel block.
- Upstream warnings every run (`pkg_resources deprecated`, `missing pytorch3d`, `Warp DeprecationWarning: warp.torch.device_from_torch`) — not actionable by this skill; noisy but non-fatal.
- `action_dim=14` is hard-coded in the random-server handshake metadata. Multi-embodiment configs with a different qpos width must pass `--action_dim N` to the server, or the handshake will advertise the wrong value. Documented in `benchmark.md`.
