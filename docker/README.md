# RoboTwin Docker dev environment

GPU dev image aligned with the root [README.md](../README.md), [`pyproject.toml`](../pyproject.toml), and [`uv.lock`](../uv.lock) (PyTorch 2.4.1 cu121, SAPIEN, `nvidia-curobo`, etc.). Two compose profiles are provided: **headless** for servers / training and **x11** for local GUI work.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) with a host NVIDIA driver matching the CUDA base (driver ≥ 525 for CUDA 12.1)

## Build

Run from anywhere; just point `-f` at a compose file in this directory.

```bash
cd /path/to/RoboTwin
docker compose -f docker/docker-compose.headless.yaml build
```

### Build args

| Arg | Default | Notes |
|-----|---------|-------|
| `CUDA_VERSION` | `12.1` | Major version of the `nvidia/cuda` base image. Must match the PyTorch cu121 wheels pinned in `uv.lock`; mismatch breaks `nvidia-curobo` compilation with a CUDA-version error. |

If you want to switch to `cu118`, edit `[[tool.uv.index]]` and `[tool.uv.sources]` in [`pyproject.toml`](../pyproject.toml), regenerate the lock file (`uv lock`), and rebuild.

First build compiles **CuRobo** and takes **≥ 20 minutes**. There is no GPU during build, so `TORCH_CUDA_ARCH_LIST` is pre-seeded with common architectures (`7.0;7.5;8.0;8.6;8.9;9.0+PTX`) for CUDA-extension compilation.

At runtime, keep the GPU device declaration in the compose file: `import envs.robot.planner` triggers **cuRobo initialising CUDA at import time**. If you drive the image with `docker run` instead of compose, add `--gpus all`.

## Mounts

| Host | Container |
|------|-----------|
| Full RoboTwin repo (`..`) | `/workspace/RoboTwin` |
| `~/.cache/huggingface` | `/root/.cache/huggingface` |

Asset download is handled by `script/_download_assets.sh` (invoked automatically when `ROBO_AUTO_ASSETS=1`). Full dataset setup follows the [official install doc](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

The compose files declare `graphics` and `video` GPU capabilities so the NVIDIA Container Toolkit mounts `libGLX_nvidia` and friends; without them, SAPIEN ray-tracing init fails with `failed to find a rendering device`.

## Sanity check

One-shot (no `up -d` needed):

```bash
docker compose -f docker/docker-compose.headless.yaml run --rm robotwin \
  bash -lc 'python -c "import torch; print(torch.cuda.is_available())" && python script/test_render.py'
```

Expected output: `True` plus a green `Render Well`.

## Usage

### Headless (server / training)

```bash
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it robotwin-headless bash
```

Inside the container, the working dir is `/workspace/RoboTwin`. Example:

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### X11 (local GUI)

Allow X on the host (`xhost +local:`) and then:

```bash
docker compose -f docker/docker-compose.x11.yaml up -d
docker exec -it robotwin-x11 bash
```

## Entrypoint behavior

- Sets `PATH` / `VIRTUAL_ENV` to `/opt/venv` (populated at build time by `uv sync --frozen`, including `nvidia-curobo`).
- The root `pyproject.toml` uses `[tool.uv] package = false` (dependency lock only); no `uv pip install -e .` is run unless a `setup.py` is present.
- On startup the entrypoint clears `/tmp/entrypoint_done`; once bootstrap completes (editable install + optional asset download) it `touch`es the file so external waiters (CI, smoke-test setup scripts) can poll it.
- `INSTALL_CLAUDE_CODE=1` will attempt to install the Claude Code CLI (requires network).
- `ROBO_AUTO_ASSETS=1` (default) runs `script/_download_assets.sh` if `assets/embodiments`, `assets/objects`, or `assets/background_texture` is empty.

### Regenerating `uv.lock`

`nvidia-curobo` needs `torch` etc. already installed in the build environment during lock resolution. Use a throwaway venv:

```bash
uv venv .lock-build-env --python 3.10
UV_PROJECT_ENVIRONMENT="$PWD/.lock-build-env" uv pip install \
  torch==2.4.1 torchvision setuptools wheel setuptools-scm cython numpy cmake ninja \
  --extra-index-url https://download.pytorch.org/whl/cu121
UV_PROJECT_ENVIRONMENT="$PWD/.lock-build-env" uv lock --python 3.10 --no-build-isolation
```

## Optional: Claude Code CLI

```bash
INSTALL_CLAUDE_CODE=1 docker compose -f docker/docker-compose.headless.yaml up -d
```

## Relation to the official docs

This image only packages Python and system libraries. Task configuration, control modes, and benchmark semantics follow the [RoboTwin 2.0 docs](https://robotwin-platform.github.io/doc/usage/index.html).
