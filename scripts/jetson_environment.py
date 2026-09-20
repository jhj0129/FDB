"""Read-only host inventory for reproducible Jetson deployment evidence."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = {}
    for name in ("numpy", "scipy", "mujoco", "mujoco-menagerie", "opencv-python", "torch",
                 "jax", "jaxlib", "flax", "optax", "pytest", "jsonschema", "pip", "setuptools", "wheel"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    data = {"machine": platform.machine(), "kernel": platform.release(),
            "python": sys.version, "python_executable": sys.executable,
            "packages": packages, "disk": shutil.disk_usage(".")._asdict(),
            "torch": {"file": torch.__file__, "version": torch.__version__,
                      "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                      "tensor_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}}
    for name, cmd in {
        "os": ("lsb_release", "-a"), "l4t": ("cat", "/etc/nv_tegra_release"),
        "model": ("cat", "/proc/device-tree/model"), "ram": ("free", "-b"),
        "cuda": ("nvcc", "--version"), "nvidia_smi": ("nvidia-smi",),
        "gcc": ("gcc", "--version"), "cmake": ("cmake", "--version"),
        "power": ("nvpmodel", "-q"),
        "jetpack": ("dpkg-query", "-W", "nvidia-jetpack"),
        "nvidia_packages": ("dpkg-query", "-W", "nvidia-l4t-core", "libcudnn9-cuda-12", "libnvinfer10"),
        "git_head": ("git", "rev-parse", "HEAD"), "git_status": ("git", "status", "--short"),
    }.items():
        data[name] = command(*cmd)
    data["cooling_types"] = {p.parent.name: p.read_text().strip() for p in Path("/sys/class/thermal").glob("cooling_device*/type")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
