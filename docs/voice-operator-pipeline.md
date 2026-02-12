# CapSeal Voice Operator Pipeline — Full Spec (Layers 1–3)

> Last updated: 2026-02-12  
> Owner: CapSeal core team  
> Scope: end-to-end path from gated OpenClaw actions to live voice operations via PersonaPlex.

## Prerequisite
- Complete `capseal-fix-spec.md` Tier 1/2 foundations first.
- This spec assumes the 4 upgrades are already landed:
  - `src/capseal/risk_engine.py` (canonical risk math)
  - `src/capseal/mcp_responses.py` (structured MCP JSON)
  - `src/capseal/risk_labels.py` (human-readable labels)
  - `src/capseal/quickstart.py` (`capseal quickstart`)

## Current Baseline (already in repo)
- MCP server: `src/capseal/mcp_server.py` with 5 tools:
  - `capseal_gate`, `capseal_record`, `capseal_seal`, `capseal_status`, `capseal_context`
- Operator daemon: `src/capseal/operator/daemon.py`
- Voice TTS: `src/capseal/operator/voice.py`
- Live voice call manager: `src/capseal/operator/voice_call.py`
- NLP command parsing: `src/capseal/operator/nlp_commands.py`
- Intervention path: `src/capseal/operator/intervention.py` + MCP `_read_intervention(...)`
- New operator readiness/provisioning:
  - `capseal operator --verify`
  - `capseal operator --provision ...`
  - `src/capseal/operator/ops.py`

## Implementation Status (2026-02-12)
- Done: Moshi binary protocol support in `src/capseal/operator/voice_call.py`
  - Connects to `/api/chat`
  - Handles handshake (`0x00`), audio (`0x01`), text tokens (`0x02`)
  - Uses offline `espeak + ffmpeg` text-to-opus uplink for event narration
- Done: Layer 3A acceptance evidence captured
  - Operator connected to live pod
  - Real MCP `gate` event triggered
  - Voice uplink sent and server audio frames received
- Done: voice command application path in daemon
  - transcript -> NLP parse -> intervention write -> gate override
  - implemented in `src/capseal/operator/daemon.py`
- Done: `capseal ops` CLI controls
  - `capseal ops start|stop|status|teardown`
  - implemented in `src/capseal/cli/ops_cmd.py`
- Done: daemon idle lifecycle
  - auto-stop after idle and session-start resume/reconnect
  - implemented in `src/capseal/operator/daemon.py` + `src/capseal/operator/runpod_ops.py`
- Remaining validation
  - true microphone-driven, user-spoken command capture on the live call has not been end-to-end validated yet

## Architecture Overview
```text
OpenClaw/Clawdbot -> MCP (gate/record/seal/status/context) -> CapSeal MCP server
                                                           -> .capseal/events.jsonl
                                                           -> Operator daemon
                                                           -> Telegram/WhatsApp/iMessage
                                                           -> PersonaPlex voice (RunPod)

Voice commands -> voice_call transcript -> nlp_commands -> intervention.json
               -> next MCP gate override -> event emit -> voice confirmation
```

---

## Layer 1: CapSeal Gates Clawdbot

### Goal
Every code-modifying agent action follows `gate -> execute -> record -> seal`.

### 1A. Skill-level enforcement
Update CapSeal OpenClaw skill instructions to require gate-before-execute.

Required behavior block for skill docs:
```markdown
When CapSeal MCP is available, always:
1. call capseal_gate before code-modifying action
2. obey decision:
   - approve: proceed
   - deny: do not execute
   - flag: proceed with warning
3. call capseal_record after execution
4. call capseal_seal at session end
Never skip gating.
```

### 1B. Framework-level enforcement
Add CapSeal MCP server to OpenClaw MCP config:
```json
{
  "capseal": {
    "command": "capseal",
    "args": ["mcp-serve", "-w", "/path/to/workspace"],
    "transport": "stdio"
  }
}
```

Discovery commands:
```bash
find ~/.openclaw -name "*.json" | xargs grep -l "mcp\\|mcporter\\|command" 2>/dev/null
ls -la ~/.openclaw/workspace/
```

### 1C. Verification sequence
```bash
# terminal 1
capseal mcp-serve -w ~/capseal

# terminal 2
capseal operator ~/capseal

# terminal 3
# trigger agent action via OpenClaw/Clawdbot
```

Expected:
- `gate` and `record` events in `.capseal/events.jsonl`
- operator notifications fire
- session seals into `.cap`
- `capseal verify` passes

Layer 1 acceptance:
- [ ] OpenClaw can call all 5 CapSeal MCP tools
- [ ] Skill text explicitly requires gate-before-execute
- [ ] One full gated session completed and sealed
- [ ] Operator notifications confirmed for gate events
- [ ] Receipt verification passes

---

## Layer 2: Clawdbot Provisions PersonaPlex on RunPod

### Goal
Clawdbot provisions GPU voice backend while every step is CapSeal-gated.

### Prereqs
- `RUNPOD_API_KEY`
- `HF_TOKEN` (with required model license accepted)

### Script target
- Create: `scripts/provision_personaplex.py`
- Responsibilities:
  - create RunPod pod (`NVIDIA A40` recommended)
  - install PersonaPlex/Moshi dependencies
  - start WebSocket service (default `8998`)
  - write resulting endpoint into operator config

### Integration with operator config
Write/update `.capseal/operator.json` voice section:
```json
{
  "voice": {
    "enabled": true,
    "provider": "personaplex",
    "live_call": true,
    "personaplex_ws_url": "wss://<pod-id>-8998.proxy.runpod.net",
    "voice_preset": "NATM1",
    "speak_gate_events": true,
    "speak_gate_decisions": ["deny", "flag"],
    "speak_min_score": 0.55
  }
}
```

### OpenClaw provisioning skill
Create:
- `skills/provision-personaplex/SKILL.md`

Must include:
- required envs (`RUNPOD_API_KEY`, `HF_TOKEN`)
- commands to run provisioning script
- connectivity verification
- cost controls (stop/resume/terminate pod)

### Layer 2 acceptance
- [ ] `scripts/provision_personaplex.py` exists and runs
- [ ] RunPod pod created with expected GPU
- [ ] PersonaPlex WebSocket is reachable
- [ ] operator config updated with WS endpoint
- [ ] provisioning session has gate/record/seal events
- [ ] resulting `.cap` verifies

---

## Layer 3: CapSeal Ops Goes Live

### Goal
Operator narrates significant gate events live and accepts voice commands for intervention.

### 3A (milestone): Voice output
Priority behavior:
- denied/flagged gate events are spoken aloud
- narration uses existing `voice_text` from composer

Current implementation path:
- Decision forcing logic: `src/capseal/operator/daemon.py` (`_should_force_gate_voice`)
- Broadcast path: `src/capseal/operator/daemon.py` (`_broadcast(...)`)
- Config knobs: `src/capseal/operator/config.py`:
  - `voice.speak_gate_events`
  - `voice.speak_gate_decisions`
  - `voice.speak_min_score`

Validation:
```bash
capseal operator . --provision --voice --voice-provider openai --telegram-chat-id <id>
capseal operator . --verify
capseal operator . --test
```

### 3B (stretch): Voice input loop
Required chain:
`voice_call transcript -> nlp_commands -> intervention.json -> next gate override`

Files:
- `src/capseal/operator/voice_call.py`
- `src/capseal/operator/nlp_commands.py`
- `src/capseal/operator/intervention.py`
- `src/capseal/mcp_server.py` (`_read_intervention`)

### 3C (cut if behind): Auto-stop + ops CLI
Add later if schedule allows:
- idle pod stop/resume
- cost reporting
- `capseal ops start|stop|status|teardown`

### Protocol caveat (critical)
`voice_call.py` assumes a text+json websocket control flow.  
PersonaPlex/Moshi protocol may require mixed binary/audio framing and exact handshake semantics.

Before productionizing 3B:
1. verify actual PersonaPlex wire protocol
2. adapt `voice_call.py` framing if needed
3. add integration test against live/staging endpoint

### Layer 3 acceptance
- [ ] operator connects to PersonaPlex WS endpoint
- [ ] denied/flagged gate events are spoken live
- [ ] voice command can trigger intervention override (stretch)
- [ ] override loop verified end-to-end (stretch)
- [ ] cost controls available (cuttable)

---

## Execution Plan (10-hour constraint)

### Priority order
1. Layer 1 verification
2. Layer 2 provisioning
3. Layer 3A voice output
4. Layer 3B stretch only if time remains
5. Layer 3C cut if behind

### Time box
- Layer 1: 0.5h
- Layer 2: 3–4h
- Layer 3A: 2–3h
- Layer 3B: 1.5–2h (stretch)
- Layer 3C: defer unless ahead

---

## Demo Script (Go-to-Market)

1. Ask clawdbot to provision PersonaPlex.
2. Show gate/record events streaming live in TUI and operator notifications.
3. Bring voice online: “CapSeal Ops online. Monitoring your session.”
4. Trigger normal action (approved narration).
5. Trigger risky action (blocked narration).
6. Issue override voice/text command.
7. Seal session and verify `.cap`.

Tagline:
`every ai action: gated, recorded, verified, narrated`

---

## Cost Reference

| GPU | VRAM | Approx $/hr | Suitability |
|---|---:|---:|---|
| A40 | 48GB | ~0.35 | Recommended baseline |
| L4 | 24GB | ~0.58 | Works, tighter headroom |
| RTX 4090 | 24GB | ~0.69 | Works, less predictable capacity |
| A100 80GB | 80GB | ~1.39 | Overkill for single-session demos |

Recommended starting point:
- A40
- enforce stop/resume discipline
- add auto-stop once 3A is stable
