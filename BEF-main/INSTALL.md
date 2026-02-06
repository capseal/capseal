# CapSeal Installation Guide

## Quick Install

### Using pipx (Recommended)

```bash
# Install pipx if you don't have it
pip install pipx
pipx ensurepath

# Install capseal
pipx install capseal

# Verify installation
capseal demo
```

### Using pip

```bash
pip install capseal
```

### Using uv (Fast)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install capseal
uv tool install capseal

# Verify installation
capseal demo
```

### Using Homebrew (macOS/Linux)

```bash
# Add the tap
brew tap capseal/tap

# Install
brew install capseal

# Verify
capseal demo
```

### From Source

```bash
git clone https://github.com/capseal/capseal.git
cd capseal
pip install -e .
capseal demo
```

## Optional Dependencies

### Cryptographic Signatures (secp256k1)

```bash
pip install capseal[crypto]
```

Note: `coincurve` requires Python < 3.14.

### Server Components

```bash
pip install capseal[server]
```

### Cloud Storage (S3/R2)

```bash
pip install capseal[cloud]
```

### AI/Agent Integration

```bash
pip install capseal[agent]
```

### Full Installation

```bash
pip install capseal[all]
```

## Sandbox Backends

CapSeal supports multiple sandbox backends for isolated execution:

### Linux

**Bubblewrap (Recommended)**
```bash
# Arch Linux
sudo pacman -S bubblewrap

# Ubuntu/Debian
sudo apt install bubblewrap

# Fedora
sudo dnf install bubblewrap
```

**Firejail**
```bash
sudo apt install firejail
```

**nsjail**
```bash
# Build from source: https://github.com/google/nsjail
```

### macOS

macOS uses `sandbox-exec` which is built-in. No additional installation required.

### Windows

Use WSL2 with a Linux sandbox backend, or Docker.

## Verify Installation

```bash
# Quick self-test
capseal demo

# Full happy-path verification
capseal init
capseal demo -o .capseal/receipts/test.json
capseal explain .capseal/receipts/test.json

# Check environment
capseal doctor .capseal/receipts/test.json
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CAPSEAL_HOME` | Global config directory | `~/.capseal` |
| `CAPSEAL_WORKSPACE` | Project workspace | `.capseal` |
| `CAPSEAL_BACKEND` | Force sandbox backend | auto-detect |
| `CAPSEAL_POLICY` | Default policy file | none |

## Troubleshooting

### "No sandbox available"

Install a sandbox backend (see above). On macOS, this warning can be ignored
as `sandbox-exec` is used automatically.

### coincurve installation fails

This typically happens on Python 3.14+. Use:
```bash
pip install capseal  # Without crypto extras
```

### Permission denied

Sandbox backends may require specific permissions:
- `bubblewrap`: User namespaces must be enabled
- `firejail`: May need root on some systems
- `nsjail`: Requires specific kernel capabilities

## Updating

```bash
# pipx
pipx upgrade capseal

# pip
pip install --upgrade capseal

# homebrew
brew upgrade capseal
```
