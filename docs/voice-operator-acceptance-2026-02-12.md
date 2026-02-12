# Voice Operator Acceptance — 2026-02-12

## Scope
- Validation 1: Layer 3A milestone (operator voice output over PersonaPlex/Moshi).
- Validation 2: Layer 3B command path (transcript -> intervention -> gate override).
- Validation 3: RunPod lifecycle controls (`capseal ops status|stop|start`) with real pod state change.

## Environment
- Pod ID: `bpmo957jy8rikt`
- Endpoint: `wss://bpmo957jy8rikt-8898.proxy.runpod.net/api/chat`
- GPU: `A40`

## Evidence: Layer 3A (Live Voice Output)
Command executed:
- Start operator daemon with voice live call enabled.
- Trigger real MCP gate event via `capseal.mcp_server._handle_gate(...)`.

Observed log evidence:
```text
[voice_call] connected (moshi_binary)
[operator] Event: gate | score=0.85 | decision=denied | files=['src/auth.py'] | p_fail=0.5
[voice_call] sent audio prompt (117636 bytes)
[voice_call] recv audio frames=10 bytes=2883
[voice_call] recv audio frames=20 bytes=6030
[operator] Voice transcript: Hi, how is it going?
```

Pass criteria met:
- voice websocket connected
- gate event triggered and processed
- operator sent audio uplink prompt
- server returned audio frames

Re-run evidence (post-resume environment):
```text
[voice_call] connected (moshi_binary)
[operator] Event: gate | score=0.60 | decision=flagged | files=['migrations/20260212_drop_table.py'] | p_fail=0.5
[voice_call] sent audio prompt (135978 bytes)
[voice_call] recv audio frames=10 bytes=1315
...
[voice_call] recv audio frames=110 bytes=16888
```

## Evidence: Layer 3B (Command -> Intervention -> Override)
Programmatic command loop test:
- call `OperatorDaemon._handle_voice_transcript("approve it")`
- verify `.capseal/intervention.json` contains `{"action":"approve"}`
- run `_handle_gate(...)` and confirm override decision

Observed result:
```text
[operator] Voice transcript: approve it
[intervention] Wrote: approve
gate_decision= approve
gate_summary= CAPSEAL GATE: APPROVED | p_fail=0.50 | simple + single-file + behavioral + untested
```

Pass criteria met:
- transcript parsed to approve command
- intervention file written
- next gate call consumed intervention and returned approve

## Evidence: Validation 3 (`capseal ops` lifecycle)
Commands executed against pod `bpmo957jy8rikt`:
- `capseal ops status --json` -> `status=RUNNING`, `network_ready=true`
- `capseal ops stop` -> follow-up `status=EXITED`
- `capseal ops start` -> follow-up polling reached `status=RUNNING`, `network_ready=true`

Bug fixed during validation:
- `src/capseal/operator/runpod_ops.py`: `resume_pod(...)` now supports SDK variants requiring `gpu_count` (fallback call `resume_pod(pod_id, 1)` on `TypeError`).

Pass criteria met:
- status reflects real pod state
- stop/start transitions confirmed on live RunPod resource
- start command is functional with current RunPod SDK signature

## Notes
- The three remaining validations requested for launch-readiness are complete.
- Stretch item still optional: true microphone-driven live speech capture (as opposed to synthetic transcript injection).
