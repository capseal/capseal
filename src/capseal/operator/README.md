# CapSeal Operator Agent

Real-time notifications for autonomous AI agent sessions.

Your agent works. You walk away. Your phone tells you what's happening.

## Quick Start

```bash
# 1. Set up Telegram bot
python daemon.py --setup telegram

# 2. Test it
python daemon.py --workspace /your/project --test

# 3. Run the operator (Terminal 1)
python daemon.py --workspace /your/project

# 4. Simulate events to test (Terminal 2)
python simulate.py --workspace /your/project --scenario full_session
```

## How It Works

```
CapSeal events.jsonl → Operator Daemon → Significance Filter → Telegram/Voice
```

The operator tails `events.jsonl` from your CapSeal session, scores each event
for significance (0.0-1.0), and sends notifications through your configured
channels when events cross the threshold.

## Notification Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0.0-0.3 | Silent | Routine, no notification |
| 0.3-0.5 | Log | Console only |
| 0.5-0.7 | Text | Telegram text message |
| 0.7-0.9 | Voice | Voice note via messaging |
| 0.9-1.0 | Critical | Live voice alert |

## Configuration

Config is loaded from (in order):
1. `--config /path/to/config.json`
2. `.capseal/operator.json` in workspace
3. `~/.capseal/operator.json`
4. Environment variables

### Environment Variables

```bash
export CAPSEAL_TELEGRAM_TOKEN="your-bot-token"
export CAPSEAL_TELEGRAM_CHAT_ID="your-chat-id"
export CAPSEAL_NOTIFY_THRESHOLD="0.5"
```

Recommended (preferred key name):

```bash
export CAPSEAL_TELEGRAM_BOT_TOKEN="your-bot-token"
```

### Config File

```json
{
  "notify_threshold": 0.5,
  "channels": {
    "telegram": {
      "bot_token_env": "CAPSEAL_TELEGRAM_BOT_TOKEN",
      "chat_id": "...",
      "voice_notes": false,
      "decision_buttons": true
    }
  }
}
```

`bot_token` is still supported for local-only setups, but storing plaintext secrets
in `operator.json` is discouraged.

## Telegram Commands

Reply to the bot in Telegram:

| Command | Action |
|---------|--------|
| `/status` | Session summary |
| `/trust` | Current trust score |
| `/files` | Files touched this session |
| `/help` | List commands |

Inline buttons on denial notifications:
- ✅ Approve — override the denial
- 🛑 Keep Blocking — confirm the denial
- 📋 Show Diff — see what the agent tried to change

## Simulator

Test without a live agent:

```bash
# Full session (approvals, denials, seal)
python simulate.py -w . --scenario full_session

# Chain break (critical alert)
python simulate.py -w . --scenario chain_break

# Sensitive file access
python simulate.py -w . --scenario sensitive_files

# Single events
python simulate.py -w . --event denial
python simulate.py -w . --event chain_break
python simulate.py -w . --event session_start

# Fast mode (0.5s between events)
python simulate.py -w . --scenario full_session --delay 0.5

# Clear old events first
python simulate.py -w . --clear --scenario full_session
```

## Architecture

```
capseal-operator/
├── daemon.py         # Main daemon, event loop
├── significance.py   # Event significance scoring (0.0-1.0)
├── composer.py       # Natural language message generation
├── config.py         # Configuration loading
├── intervention.py   # Write commands back to MCP server
├── simulate.py       # Event simulator for testing
├── channels/
│   └── telegram.py   # Telegram Bot API channel
└── voice.py          # PersonaPlex integration (Phase 5)
```

## Roadmap

- [x] Phase 1-3: Telegram text + significance filter + message composer
- [ ] Phase 4: Intervention channel (Telegram approve/deny buttons wired to MCP)
- [ ] Phase 5: PersonaPlex voice notes
- [ ] Phase 6: Live full-duplex voice channel
- [ ] Phase 7: WhatsApp + iMessage channels
- [ ] Phase 8: LLM-powered natural language command parsing
