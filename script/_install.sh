#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Installing dependencies from pyproject.toml / uv.lock (uv sync) ..."
uv sync

echo "Installing pytorch3d (optional sim extra; not in lock) ..."
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git"

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION="$(uv pip show sapien | grep '^Location:' | awk '{print $2}')/sapien"
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"

echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION="$(uv pip show mplib | grep '^Location:' | awk '{print $2}')/mplib"
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"

echo "Installation basic environment complete!"
echo "CuRobo (nvidia-curobo) is installed via uv.lock; no envs/curobo clone needed."
echo "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See official RoboTwin documentation for more instructions."
