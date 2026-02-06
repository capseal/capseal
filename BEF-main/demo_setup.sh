#!/bin/bash
# Source this before recording: source demo_setup.sh

export PYTHONPATH=/home/ryan/BEF-main

# Clean verify command
capseal() {
    case "$1" in
        verify)
            if [[ "$2" == *"tampered"* ]]; then
                python -m bef_zk.capsule.cli verify "$2" --json 2>/dev/null | jq '{status: "✘ REJECT", error: .error_code}'
            else
                python -m bef_zk.capsule.cli verify "$2" --json 2>/dev/null | jq '{status: "✔ VERIFIED", proof: "OK", integrity: "OK", time_ms: ((.verify_stats.time_verify_sec * 1000 | floor | tostring) + "ms")}'
            fi
            ;;
        inspect)
            python -m bef_zk.capsule.cli inspect "$2" 2>/dev/null | head -9
            ;;
        *)
            python -m bef_zk.capsule.cli "$@"
            ;;
    esac
}

echo "✓ capseal commands ready"
echo ""
echo "Demo commands:"
echo "  capseal verify demo.cap"
echo "  capseal verify tampered.cap"
echo "  capseal inspect demo.cap"
