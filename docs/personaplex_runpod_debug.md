# PersonaPlex RunPod Template Debug

## Why this exists
When pods show `desiredStatus=RUNNING` but `runtime.ports=null` and `uptimeSeconds=0`, the container template is usually not booting into a reachable runtime.

## Manual console sanity check (one pod only)
1. In RunPod web console, create a single pod.
2. Use image: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04`
3. Expose ports: `8998/http,22/tcp`
4. Enable public IP/SSH.
5. Confirm pod shows populated runtime ports and SSH connect command.

If this manual pod gets ports, script config was the issue.
If this manual pod also has null ports, account/region networking is the issue.

## Script defaults now
- Default image switched to `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04`
- Public IP requested via `support_public_ip=true`
- Cloud mode explicit (`COMMUNITY` by default)
- Fallback GPUs are empty by default (no automatic cost-burning churn)

## Safe script usage (single-attempt)
```bash
RUNPOD_API_KEY=... HF_TOKEN=... \
python scripts/provision_personaplex.py \
  --workspace /home/ryan/capseal \
  --gpu "NVIDIA A40" \
  --fallback-gpus "" \
  --gpu-ready-timeout-seconds 180
```

## Reuse a manually-created pod
```bash
RUNPOD_API_KEY=... HF_TOKEN=... \
python scripts/provision_personaplex.py \
  --workspace /home/ryan/capseal \
  --existing-pod-id <POD_ID> \
  --fallback-gpus ""
```

This skips pod creation and continues with install/start/verify on that pod.
