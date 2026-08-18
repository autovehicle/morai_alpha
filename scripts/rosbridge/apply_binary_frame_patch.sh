#!/usr/bin/env bash
# Applies patch_protocol_incoming.py to whichever rosbridge_library package
# is actually live (respects catkin workspace overlays, same as
# apply_resilience_patch.sh). Only escalates to sudo if the resolved file
# isn't writable by the current user.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -ge 1 ]; then
    TARGET="$1"
    shift
else
    TARGET="$(python3 -c 'import rosbridge_library.protocol as m; print(m.__file__)' | tail -n 1)"
fi

echo "target: $TARGET"

if [ -w "$TARGET" ]; then
    python3 "$SCRIPT_DIR/patch_protocol_incoming.py" "$TARGET" "$@"
else
    sudo python3 "$SCRIPT_DIR/patch_protocol_incoming.py" "$TARGET" "$@"
fi
