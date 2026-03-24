#!/bin/bash
set -e

export PATH="/opt/venv/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export VIRTUAL_ENV="/opt/venv"

# Root pyproject uses [tool.uv] package = false (dependency lock only). Skip editable install.
if [ -f "/workspace/RoboTwin/setup.py" ]; then
    echo ">> Installing editable package (setup.py)..."
    cd /workspace/RoboTwin && uv pip install -e . --no-deps && cd - > /dev/null
fi

if [ "${INSTALL_CLAUDE_CODE}" = "1" ]; then
    echo ">> Installing Claude Code CLI..."
    if curl -fsSL https://claude.ai/install.sh | bash 2>/dev/null; then
        export PATH="${HOME}/.local/bin:${PATH}"
        echo ">> Claude Code CLI installed. Run 'claude' to start."
    else
        echo ">> Claude Code CLI install skipped (network unavailable)."
    fi
fi

RT_ROOT="/workspace/RoboTwin"
ROBO_AUTO_ASSETS="${ROBO_AUTO_ASSETS:-1}"

if [ "$ROBO_AUTO_ASSETS" = "1" ] && [ -d "$RT_ROOT" ]; then
    _rt_dir_nonempty() {
        local d="$1"
        [ -d "$d" ] && [ -n "$(find "$d" -mindepth 1 -print -quit 2>/dev/null)" ]
    }
    if _rt_dir_nonempty "$RT_ROOT/assets/embodiments" \
        && _rt_dir_nonempty "$RT_ROOT/assets/objects" \
        && _rt_dir_nonempty "$RT_ROOT/assets/background_texture"; then
        echo ">> RoboTwin assets already present (embodiments/, objects/, background_texture/); skipping download."
    else
        echo ">> RoboTwin assets missing or incomplete; running script/_download_assets.sh (writes to bind-mounted repo on host)..."
        (cd "$RT_ROOT" && bash script/_download_assets.sh)
    fi
elif [ "$ROBO_AUTO_ASSETS" != "1" ]; then
    echo ">> ROBO_AUTO_ASSETS=${ROBO_AUTO_ASSETS}: automatic assets download disabled."
fi

echo ">> Ready. Repo: $RT_ROOT"
exec "$@"
