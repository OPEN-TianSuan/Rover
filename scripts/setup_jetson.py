"""Prepare the pinned JetPack 6 / CUDA 12.6 runtime, then sync using uv.

Run with system Python: python3 scripts/setup_jetson.py
Artifacts already present in vendor/jetson are reused after SHA-256 verification.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/jetson"
WHEELS = [
    (
        "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/46b/b8b13f844b211/torch-2.11.0-cp310-cp310-linux_aarch64.whl",
        "46bb8b13f844b2111f6f41abc703d01075a7185e0c587cb002e957ede2b8c8a3",
    ),
    (
        "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d11/6f08d3d62417d/torchvision-0.26.0-cp310-cp310-linux_aarch64.whl",
        "d116f08d3d62417dc36cbf41923e491b8b49b77399575f0878ae391e6e77a159",
    ),
]
CUDSS_URL = (
    "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/"
    "linux-aarch64/libcudss-linux-aarch64-0.8.0.10_cuda12-archive.tar.xz"
)
CUDSS_SHA256 = "11cf97c8cecf3a651bab1890894c8b8793666a17ad3002104e573efa564af231"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, expected):
    path = VENDOR / url.rsplit("/", 1)[1]
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {path.name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if checksum(temporary) != expected:
            raise RuntimeError(f"Checksum mismatch: {temporary}")
        temporary.replace(path)
    if checksum(path) != expected:
        raise RuntimeError(f"Checksum mismatch: {path}")
    return path


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise SystemExit("This installer targets Jetson aarch64 with JetPack 6 / CUDA 12.6.")
    VENDOR.mkdir(parents=True, exist_ok=True)
    for url, expected in WHEELS:
        fetch(url, expected)
    library = VENDOR / "libcudss.so.0"
    if not library.exists():
        archive = fetch(CUDSS_URL, CUDSS_SHA256)
        with tarfile.open(archive, "r:xz") as tar:
            # Extract only these known regular files, without trusting archive paths.
            for suffix, target in [("/lib/libcudss.so.0.8.0", library), ("/LICENSE", VENDOR / "LICENSE-cuDSS")]:
                member = next(m for m in tar.getmembers() if m.name.endswith(suffix) and m.isfile())
                with tar.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    if checksum(library) != "b2f016af586fd64a22b8195aa1cec5e1e7e2022fa204dba0d8f2605f235f51f9":
        raise RuntimeError("cuDSS library checksum mismatch")
    subprocess.run(["uv", "sync", "--python", "3.10"], cwd=ROOT, check=True)
    # The Jetson wheel expects cuDSS on its ELF library search path. Install a
    # relative link alongside torch's other libraries; no global LD_LIBRARY_PATH needed.
    target = ROOT / ".venv/lib/python3.10/site-packages/torch/lib/libcudss.so.0"
    if not target.exists():
        target.symlink_to(os.path.relpath(library, target.parent))
    print("Rover is installed. Run uv run rover --help to get started.")


if __name__ == "__main__":
    main()
