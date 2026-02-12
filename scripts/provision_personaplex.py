#!/usr/bin/env python3
"""
Provision PersonaPlex (Moshi) on RunPod for CapSeal Ops.

Primary flow:
1) Create pod (A40 by default, fallback GPUs optional)
2) SSH in and install runtime + models
3) Start Moshi server on :8998
4) Verify WebSocket reachability from this machine
5) Update operator config with ws endpoint and pod id

Usage:
  python scripts/provision_personaplex.py \
    --runpod-key "$RUNPOD_API_KEY" \
    --hf-token "$HF_TOKEN"
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


DEFAULT_GPU = "NVIDIA A40"
# Keep retries explicit to avoid burning credits on repeated failed allocations.
DEFAULT_FALLBACKS: list[str] = []
# Known-good RunPod image that stays alive and exposes networking/SSH reliably.
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04"
DEFAULT_PORTS = "8998/http,22/tcp"


def _log(msg: str) -> None:
    print(f"[provision] {msg}", flush=True)


def _redacted_cmd(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    secret_flags = {"--apiKey", "--runpod-key", "--hf-token"}
    i = 0
    while i < len(redacted):
        if redacted[i] in secret_flags and i + 1 < len(redacted):
            redacted[i + 1] = "***"
            i += 2
            continue
        i += 1
    return redacted


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    _log("run: " + " ".join(shlex.quote(x) for x in _redacted_cmd(cmd)))
    cp = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"command failed ({cp.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )
    return cp


def _ensure_pkg(name: str) -> None:
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass

    _log(f"installing python dependency: {name}")
    _run([sys.executable, "-m", "pip", "install", name], check=True)


def _check_hf_token(hf_token: str) -> None:
    """Validate HF token upfront to avoid long provisioning failures."""
    req = Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {hf_token}"},
    )
    try:
        with urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HF token validation failed (status={resp.status})")
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("HF token is invalid or expired (401)") from exc
        raise RuntimeError(f"HF token validation failed (HTTP {exc.code})") from exc
    except URLError as exc:
        raise RuntimeError(f"HF token validation failed (network error: {exc})") from exc


def _configure_runpodctl(api_key: str) -> None:
    # runpodctl requires persistent config (no --apiKey flag on subcommands).
    _run(["runpodctl", "config", "--apiKey", api_key], check=True)


def _extract_pod_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("id", "podId", "pod_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = _extract_pod_id(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_pod_id(item)
            if found:
                return found
    return None


def _runpod_create_pod(
    runpod_mod: Any,
    gpu: str,
    hf_token: str,
    secure_cloud: bool,
    image_name: str,
) -> str:
    create_fn = getattr(runpod_mod, "create_pod", None)
    if create_fn is None:
        raise RuntimeError("runpod.create_pod not found in installed runpod package")

    base_kwargs = {
        "name": "capseal-ops-personaplex",
        "image_name": image_name,
        "gpu_type_id": gpu,
        "cloud_type": "SECURE" if secure_cloud else "COMMUNITY",
        "support_public_ip": True,
        "ports": DEFAULT_PORTS,
        "volume_in_gb": 30,
        "container_disk_in_gb": 30,
        "start_ssh": True,
        "env": {
            "HF_TOKEN": hf_token,
            "DEBIAN_FRONTEND": "noninteractive",
        },
    }
    sig = inspect.signature(create_fn)
    kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}
    # Some runpod SDK builds print raw GraphQL responses to stdout, including env.
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        result = create_fn(**kwargs)
    pod_id = _extract_pod_id(result)
    if not pod_id:
        raise RuntimeError(f"failed to parse pod id from create_pod response: {result!r}")
    return pod_id


def _runpod_get_pod(runpod_mod: Any, pod_id: str) -> dict[str, Any]:
    get_fn = getattr(runpod_mod, "get_pod", None)
    if get_fn is None:
        raise RuntimeError("runpod.get_pod not found in installed runpod package")
    payload = get_fn(pod_id)
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"unexpected get_pod response: {payload!r}")


def _wait_for_running(runpod_mod: Any, pod_id: str, timeout_s: int) -> dict[str, Any]:
    _log(f"waiting for pod {pod_id} to reach RUNNING (timeout {timeout_s}s)")
    started = time.time()
    while True:
        pod = _runpod_get_pod(runpod_mod, pod_id)
        desired = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports")
        ports_ready = bool(ports)
        if desired == "RUNNING" and ports_ready:
            _log(f"pod {pod_id} is RUNNING")
            return pod
        if time.time() - started > timeout_s:
            raise TimeoutError(
                f"pod {pod_id} did not become network-ready in {timeout_s}s "
                f"(status={desired}, ports_ready={ports_ready})"
            )
        time.sleep(5)


def _select_http_private_port(pod: dict[str, Any], preferred: int = 8998) -> int:
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    http_private = [
        int(p.get("privatePort"))
        for p in ports
        if p.get("type") == "http" and p.get("privatePort") is not None
    ]
    if preferred in http_private:
        return preferred
    # Avoid common template-reserved web UI port when alternatives exist.
    for p in http_private:
        if p != 8888:
            return p
    if http_private:
        return http_private[0]
    return preferred


def _ws_url_for_pod(pod_id: str, private_port: int) -> str:
    return f"wss://{pod_id}-{private_port}.proxy.runpod.net"


def _get_ssh_command(pod_id: str, timeout_s: int = 600) -> list[str]:
    started = time.time()
    last_stdout = ""
    last_stderr = ""
    while True:
        cp = _run(["runpodctl", "ssh", "connect", pod_id], check=False)
        last_stdout = cp.stdout or ""
        last_stderr = cp.stderr or ""
        lines = [ln.strip() for ln in last_stdout.splitlines() if ln.strip()]
        for line in lines:
            if "ssh " in line:
                idx = line.find("ssh ")
                return shlex.split(line[idx:])
        if lines and lines[0].startswith("ssh "):
            return shlex.split(lines[0])
        if time.time() - started > timeout_s:
            break
        time.sleep(5)
    raise RuntimeError(
        "could not parse ssh command from runpodctl output before timeout:\n"
        f"{last_stdout}\n{last_stderr}"
    )


def _ssh_bash(ssh_cmd: list[str], script: str, timeout_s: int = 1800, connect_retry_s: int = 240) -> None:
    cmd = ssh_cmd + ["bash", "-lc", script]
    _log("remote ssh exec")
    deadline = time.time() + max(30, min(connect_retry_s, timeout_s))
    while True:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
        if cp.returncode == 0:
            return
        stderr = cp.stderr or ""
        transient = cp.returncode == 255 and (
            "Connection refused" in stderr
            or "Operation timed out" in stderr
            or "No route to host" in stderr
        )
        if transient and time.time() < deadline:
            _log("ssh endpoint not ready yet, retrying in 5s")
            time.sleep(5)
            continue
        raise RuntimeError(
            f"remote command failed ({cp.returncode})\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )


def _install_script() -> str:
    return r"""
set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip python3-dev \
  git curl ffmpeg libopus-dev ca-certificates build-essential openssl

python3 -m venv /opt/personaplex-env
. /opt/personaplex-env/bin/activate
python -m pip install --upgrade pip

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install moshi huggingface_hub websockets aiohttp

if [ ! -d /opt/personaplex ]; then
  git clone https://github.com/NVIDIA/personaplex.git /opt/personaplex
fi

mkdir -p /opt/models
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('kyutai/moshika-pytorch-bf16','tokenizer-e351c8d8-checkpoint125.safetensors', local_dir='/opt/models')
hf_hub_download('kyutai/moshika-pytorch-bf16','model.safetensors', local_dir='/opt/models')
hf_hub_download('kyutai/moshika-pytorch-bf16','tokenizer_spm_32k_3.model', local_dir='/opt/models')
print('model weights downloaded')
PY

cat > /opt/capseal-ops-persona.txt <<'PERSONA'
You are CapSeal Ops, a calm security operator who narrates AI-agent activity.
Be concise, technical, and direct. Prioritize risk decisions, failures, and overrides.
PERSONA
"""


def _start_script(service_port: int) -> str:
    return rf"""
set -euxo pipefail
. /opt/personaplex-env/bin/activate
cd /opt/personaplex

# If using a template-provided web UI port, stop it first.
if [ "{service_port}" = "8888" ]; then
  pkill -f "jupyter-lab" || true
  pkill -f "jupyter-notebook" || true
fi

# Stop prior server only via explicit pid file to avoid killing this shell.
if [ -f /var/run/moshi-server.pid ]; then
  OLD_PID="$(cat /var/run/moshi-server.pid || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" || true
    sleep 1
  fi
fi
PORT_ARG=""
if python -m moshi.server --help 2>&1 | grep -q -- '--port'; then
  PORT_ARG="--port {service_port}"
fi
nohup python -m moshi.server \
  --host 0.0.0.0 \
  $PORT_ARG \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  > /var/log/moshi-server.log 2>&1 &
MOSHI_PID=$!
echo "$MOSHI_PID" > /var/run/moshi-server.pid

for i in $(seq 1 40); do
  if ! kill -0 "$MOSHI_PID" 2>/dev/null; then
    echo "moshi-process-exited"
    tail -n 120 /var/log/moshi-server.log || true
    exit 1
  fi
  if ss -ltnp | grep -E ":{service_port}[[:space:]]" | grep -q "pid=$MOSHI_PID"; then
    echo "moshi-ready"
    exit 0
  fi
  sleep 3
done

echo "moshi-not-ready"
tail -n 120 /var/log/moshi-server.log || true
exit 1
"""


def _verify_ws(ws_url: str, timeout_s: int = 20) -> None:
    _ensure_pkg("websockets")
    import asyncio
    import websockets

    async def _connect() -> None:
        target = ws_url.rstrip("/") + "/api/chat"
        async with websockets.connect(target, open_timeout=timeout_s, close_timeout=5):
            return

    asyncio.run(_connect())


def _update_operator_config(ws_url: str, pod_id: str, workspace: Path | None) -> None:
    targets = [Path.home() / ".capseal" / "operator.json"]
    if workspace:
        targets.append(workspace / ".capseal" / "operator.json")

    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg: dict[str, Any] = {}
        if path.exists():
            try:
                cfg = json.loads(path.read_text())
            except json.JSONDecodeError:
                cfg = {}

        voice = cfg.setdefault("voice", {})
        voice["enabled"] = True
        voice["provider"] = "personaplex"
        voice["live_call"] = True
        voice["personaplex_ws_url"] = ws_url
        voice["voice_preset"] = "NATM1"
        voice["speak_gate_events"] = True
        voice["speak_gate_decisions"] = ["deny", "flag"]
        voice["speak_min_score"] = 0.55
        cfg["runpod_pod_id"] = pod_id

        path.write_text(json.dumps(cfg, indent=2) + "\n")
        _log(f"updated operator config: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision PersonaPlex server on RunPod for CapSeal Ops")
    parser.add_argument("--runpod-key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API key")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE, help=f"Pod image (default: {DEFAULT_IMAGE})")
    parser.add_argument("--gpu", default=DEFAULT_GPU, help=f"Primary GPU type (default: {DEFAULT_GPU})")
    parser.add_argument(
        "--fallback-gpus",
        default=",".join(DEFAULT_FALLBACKS),
        help="Comma-separated fallback GPU types used if primary fails",
    )
    parser.add_argument("--workspace", default=".", help="Workspace for .capseal/operator.json update")
    parser.add_argument("--existing-pod-id", default="", help="Reuse an already-created pod id (skip create loop)")
    parser.add_argument("--service-port", type=int, default=0, help="Force private HTTP port for Moshi (e.g., 8888)")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency/model install and only start/verify")
    parser.add_argument("--secure-cloud", action="store_true", help="Use secure cloud instead of community cloud")
    parser.add_argument(
        "--gpu-ready-timeout-seconds",
        type=int,
        default=240,
        help="Per-GPU wait for network-ready pod (runtime ports available)",
    )
    parser.add_argument("--hard-stop-minutes", type=int, default=90, help="Overall hard stop for provisioning")
    args = parser.parse_args()

    if not args.runpod_key:
        raise SystemExit("RUNPOD_API_KEY is required (--runpod-key or env)")
    if not args.hf_token:
        raise SystemExit("HF_TOKEN is required (--hf-token or env)")
    _check_hf_token(args.hf_token)

    hard_deadline = time.time() + args.hard_stop_minutes * 60
    workspace = Path(args.workspace).resolve()

    _ensure_pkg("runpod")
    runpod = importlib.import_module("runpod")
    runpod.api_key = args.runpod_key
    _configure_runpodctl(args.runpod_key)

    gpu_candidates = [args.gpu] + [g.strip() for g in args.fallback_gpus.split(",") if g.strip()]
    pod_id: str | None = None
    pod_info: dict[str, Any] | None = None

    if args.existing_pod_id:
        pod_id = args.existing_pod_id.strip()
        wait_left = max(60, int(hard_deadline - time.time()))
        pod_info = _wait_for_running(runpod, pod_id, timeout_s=min(wait_left, args.gpu_ready_timeout_seconds))
        _log(f"reusing existing pod: {pod_id}")
    else:
        for idx, gpu in enumerate(gpu_candidates, start=1):
            if time.time() > hard_deadline:
                raise TimeoutError("hard stop exceeded before pod creation completed")
            _log(f"creating pod (attempt {idx}/{len(gpu_candidates)}): gpu={gpu}")
            try:
                candidate_id = _runpod_create_pod(
                    runpod,
                    gpu,
                    args.hf_token,
                    secure_cloud=args.secure_cloud,
                    image_name=args.image_name,
                )
                wait_left = max(60, int(hard_deadline - time.time()))
                candidate_info = _wait_for_running(
                    runpod,
                    candidate_id,
                    timeout_s=min(wait_left, args.gpu_ready_timeout_seconds),
                )
                pod_id = candidate_id
                pod_info = candidate_info
                _log(f"pod ready: {pod_id} (gpu={gpu})")
                break
            except Exception as exc:
                _log(f"pod attempt failed on gpu={gpu}: {exc}")
                if idx == len(gpu_candidates):
                    raise

    if not pod_id or not pod_info:
        raise RuntimeError("failed to provision a pod on all GPU candidates")

    service_port = args.service_port if args.service_port > 0 else _select_http_private_port(pod_info, preferred=8998)
    ws_url = _ws_url_for_pod(pod_id, service_port)
    _log(f"websocket target: {ws_url}")
    _log(f"selected private service port: {service_port}")

    if time.time() > hard_deadline:
        raise TimeoutError("hard stop exceeded before install/start")
    ssh_wait_left = max(120, int(hard_deadline - time.time()))
    ssh_cmd = _get_ssh_command(pod_id, timeout_s=min(ssh_wait_left, 900))
    _log("ssh command acquired")

    if not args.skip_install:
        wait_left = max(60, int(hard_deadline - time.time()))
        _ssh_bash(ssh_cmd, _install_script(), timeout_s=min(wait_left, 3600))
        _log("personaplex dependencies installed")
    else:
        _log("skipping install as requested")

    wait_left = max(60, int(hard_deadline - time.time()))
    _ssh_bash(ssh_cmd, _start_script(service_port), timeout_s=min(wait_left, 1200))
    _log("moshi server startup command completed")

    if time.time() > hard_deadline:
        raise TimeoutError("hard stop exceeded before websocket verification")
    _verify_ws(ws_url, timeout_s=20)
    _log("websocket reachable from local machine")

    _update_operator_config(ws_url, pod_id, workspace)

    _log("done")
    print(json.dumps(
        {
            "ok": True,
            "pod_id": pod_id,
            "ws_url": ws_url,
            "stop_cmd": f"python -c \"import os,runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; runpod.stop_pod('{pod_id}')\"",
            "terminate_cmd": f"python -c \"import os,runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; runpod.terminate_pod('{pod_id}')\"",
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _log(f"failed: {exc}")
        raise
