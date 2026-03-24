# RoboTwin Docker 开发环境

基于仓库根目录 [README.md](../README.md) 与 `script/requirements.txt` 的主线依赖（PyTorch 2.4.1、SAPIEN 等），提供 GPU 开发与无头/带显示器两种 Compose 配置。

## 前置条件

- [Docker](https://docs.docker.com/engine/install/) 与 [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（宿主机需已安装匹配版本的 NVIDIA 驱动）

## 构建

在**仓库根目录**的上一级或任意位置执行时，请将 `-f` 指向本目录下的 compose 文件。

```bash
cd /path/to/RoboTwin
docker compose -f docker/docker-compose.headless.yaml build
```

### 构建参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `CUDA_VERSION` | `12.1` | 基础镜像 `nvidia/cuda` 主版本（与 `Dockerfile` 中 tag 一致） |
| `TORCH_CUDA` | `cu121` | PyTorch wheel 后缀；若驱动较旧可改为 `cu118` 并同时将 `CUDA_VERSION` 设为 `11.8`（需自行调整 `Dockerfile` 中 `FROM` 的 tag 与 cu 版本一致） |

示例：

```bash
CUDA_VERSION=11.8 TORCH_CUDA=cu118 docker compose -f docker/docker-compose.headless.yaml build
```

（使用 `cu118` 时请将 `Dockerfile` 第一行 `FROM` 改为 `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` 或你环境可用的 11.8 tag。）

## 数据与缓存挂载

| 宿主机 | 容器内 |
|--------|--------|
| 整个 RoboTwin 仓库 | `/workspace/RoboTwin` |
| `~/.cache/huggingface` | `/root/.cache/huggingface` |

完整数据与资产下载仍须按 [官方安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 在容器内或挂载目录中完成。

Compose 中为 GPU 设备声明了 `graphics` 与 `video` 能力，以便 NVIDIA Container Toolkit 挂载 `libGLX_nvidia` 等库；否则 SAPIEN 光追初始化可能报 `failed to find a rendering device`。

## 验证（可选）

一次性进入容器并跑官方渲染自检（无需先 `up -d`）：

```bash
docker compose -f docker/docker-compose.headless.yaml run --rm robotwin \
  bash -lc 'python -c "import torch; print(torch.cuda.is_available())" && python script/test_render.py'
```

期望输出包含 `True` 与绿色的 `Render Well`。

## 使用

### 无头（训练 / 服务器）

```bash
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it robotwin-headless bash
```

进入容器后工作目录为 `/workspace/RoboTwin`，例如：

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### X11（本机图形界面）

宿主机允许 X 访问（示例：`xhost +local:`），然后：

```bash
docker compose -f docker/docker-compose.x11.yaml up -d
docker exec -it robotwin-x11 bash
```

## 启动时 entrypoint 行为

- 将 `PATH` / `VIRTUAL_ENV` 指向镜像内 `/opt/venv`（已通过 `uv` 安装 `script/requirements.txt` 中的依赖）。
- 若仓库根目录将来增加 `pyproject.toml` 或 `setup.py`，会自动执行 `uv pip install -e .`。
- 若设置环境变量 `INSTALL_CLAUDE_CODE=1`，会尝试安装 Claude Code CLI（需网络）。

## 可选：启用 Claude Code CLI

```bash
INSTALL_CLAUDE_CODE=1 docker compose -f docker/docker-compose.headless.yaml up -d
```

## 与官方文档的关系

容器仅封装 Python 与系统库；任务配置、控制模式与基准说明仍以 [RoboTwin 2.0 文档](https://robotwin-platform.github.io/doc/usage/index.html) 为准。
