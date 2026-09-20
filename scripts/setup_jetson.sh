#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(uname -m)" = aarch64 || { echo 'This setup is for aarch64 Jetson.' >&2; exit 1; }
cat /etc/nv_tegra_release
df -h .
free -h
# Reuse the working host wheel; installing a generic ARM torch can disable CUDA.
python3 - <<'PY'
import sys
import torch
assert sys.version_info[:2] == (3, 10), 'This lock was validated with Python 3.10'
assert torch.cuda.is_available(), 'Install a matching Jetson CUDA torch before setup'
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('Host CUDA PASS', torch.__version__, torch.version.cuda, torch.__file__)
PY
if [[ ! -e .venv ]]; then
    python3 -m venv --system-site-packages .venv
fi
test -x .venv/bin/python || { echo 'Existing .venv is not a usable venv.' >&2; exit 1; }
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
.venv/bin/python -m pip install 'pip==25.3' 'setuptools==79.0.1' 'wheel==0.47.0'
.venv/bin/python -m pip install -r requirements-jetson.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -m pip check
MUJOCO_GL=egl JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
import numpy as np
import cv2
import mujoco
import torch
import fdb.core
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
with mujoco.Renderer(m, width=64, height=64) as renderer:
    renderer.update_scene(d)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
assert rgb.shape == (64, 64, 3) and np.isfinite(depth).all()
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('FDB, CUDA, MuJoCo EGL PASS', np.__version__, cv2.__version__, mujoco.__version__)
PY
