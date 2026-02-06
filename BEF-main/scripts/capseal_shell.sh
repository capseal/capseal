#!/bin/bash
# Capseal Interactive Shell - sandboxed environment for diff review workflows
# Usage: ./scripts/capseal_shell.sh [repo_a] [repo_b]

set -e

REPO_A="${1:-/home/ryan/projects/CapsuleTech}"
REPO_B="${2:-/home/ryan/BEF-main}"
WORKSPACE="/home/ryan/BEF-main"
SANDBOX_DIR="$WORKSPACE/.capseal/sandbox_session_$(date +%s)"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           CapSeal Interactive Review Shell                 ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Repo A: $REPO_A"
echo "║  Repo B: $REPO_B"
echo "║  Sandbox: $SANDBOX_DIR"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Create sandbox directory
mkdir -p "$SANDBOX_DIR"/{diffs,reviews,capsules}

# Activate venv
source "$WORKSPACE/.venv/bin/activate"
export PYTHONPATH="$WORKSPACE"
export CAPSEAL_WORKSPACE_ROOT="$WORKSPACE"
export PATH="$HOME/.local/node_modules/.bin:$WORKSPACE:$PATH"

# Generate initial diff
echo "[1/3] Generating diff..."
diff -rq "$REPO_A" "$REPO_B" \
  --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' \
  --exclude='.git' --exclude='*.pyc' --exclude='.capseal' \
  > "$SANDBOX_DIR/diffs/summary.txt" 2>/dev/null || true

# Count changes
DIFF_COUNT=$(wc -l < "$SANDBOX_DIR/diffs/summary.txt")
echo "   Found $DIFF_COUNT differences"

# Generate detailed diff for common files
echo "[2/3] Generating detailed patches..."
comm -12 \
  <(cd "$REPO_A" && find . -type f -name "*.py" | sort) \
  <(cd "$REPO_B" && find . -type f -name "*.py" | sort) 2>/dev/null | \
while read -r file; do
  if [ -f "$REPO_A/$file" ] && [ -f "$REPO_B/$file" ]; then
    diff -u "$REPO_A/$file" "$REPO_B/$file" >> "$SANDBOX_DIR/diffs/patches.diff" 2>/dev/null || true
  fi
done

PATCH_LINES=$(wc -l < "$SANDBOX_DIR/diffs/patches.diff" 2>/dev/null || echo "0")
echo "   Generated $PATCH_LINES lines of patches"

# Create session metadata
cat > "$SANDBOX_DIR/session.json" << EOF
{
  "session_id": "$(basename $SANDBOX_DIR)",
  "created_at": "$(date -Iseconds)",
  "repo_a": "$REPO_A",
  "repo_b": "$REPO_B",
  "diff_count": $DIFF_COUNT,
  "patch_lines": $PATCH_LINES,
  "status": "ready"
}
EOF

echo "[3/3] Starting interactive shell..."
echo ""
echo "Available commands:"
echo "  cline \"<prompt>\"     - Ask Cline to review/act"
echo "  capseal greptile query \"<q>\" -r owner/repo  - Query codebase"
echo "  capseal verify <capsule>     - Verify a capsule"
echo "  cat \$DIFF            - View diff summary"
echo "  cat \$PATCHES         - View detailed patches"
echo "  exit                 - Exit shell"
echo ""

# Export convenience variables
export DIFF="$SANDBOX_DIR/diffs/summary.txt"
export PATCHES="$SANDBOX_DIR/diffs/patches.diff"
export SESSION="$SANDBOX_DIR/session.json"
export REVIEWS="$SANDBOX_DIR/reviews"
export CAPSULES="$SANDBOX_DIR/capsules"
export PS1="[capseal-review] \w \$ "

# Start interactive shell
exec bash --norc --noprofile -i
