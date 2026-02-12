---
name: provision-personaplex
description: >
  Provision a PersonaPlex (NVIDIA Moshi) voice server on RunPod for CapSeal Ops.
  Creates a GPU pod, installs PersonaPlex, starts Moshi on port 8998, verifies
  WebSocket reachability, and writes operator voice config.
emoji: 🎙️
metadata:
  openclaw:
    requires:
      bins: ["python3", "capseal", "runpodctl", "ssh"]
      env: ["RUNPOD_API_KEY", "HF_TOKEN"]
---

# Provision PersonaPlex For CapSeal Ops

## Prerequisites
- `RUNPOD_API_KEY`
- `HF_TOKEN` (license accepted for required model repos)

## Steps

### 1. Run provisioning script

```bash
python scripts/provision_personaplex.py \
  --runpod-key "$RUNPOD_API_KEY" \
  --hf-token "$HF_TOKEN" \
  --gpu "NVIDIA A40" \
  --fallback-gpus "NVIDIA L4,NVIDIA RTX 4090" \
  --hard-stop-minutes 90
```

### 2. Verify WS endpoint

The script prints:
- `pod_id`
- `ws_url`

If it exits with `ok: true`, WS reachability was verified from this machine.

### 3. Verify operator config

```bash
jq '.voice' ~/.capseal/operator.json
```

Expected:
- `provider = "personaplex"`
- `live_call = true`
- `personaplex_ws_url = "wss://<pod-id>-8998.proxy.runpod.net"`

## Cost controls

Use printed commands to stop or terminate:
- stop pod
- terminate pod

## Troubleshooting
- If pod creation fails on A40, script automatically tries fallback GPUs.
- If install/start fails, inspect remote logs:
  - `/var/log/moshi-server.log`
- If WS verification fails, check pod port mapping and server readiness.

