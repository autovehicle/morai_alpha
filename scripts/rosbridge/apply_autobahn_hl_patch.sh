#!/usr/bin/env bash
# Applies patch_autobahn_hl.py to whichever autobahn package is actually
# live. Only escalates to sudo if the resolved file isn't writable by the
# current user (the pip --user install lives under the user's home, so
# this normally runs without sudo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -ge 1 ]; then
    TARGET="$1"
    shift
else
    TARGET="$(python3 -c 'import autobahn.util as m; print(m.__file__)' | tail -n 1)"
fi

echo "target: $TARGET"

if [ -w "$TARGET" ]; then
    python3 "$SCRIPT_DIR/patch_autobahn_hl.py" "$TARGET" "$@"
else
    sudo python3 "$SCRIPT_DIR/patch_autobahn_hl.py" "$TARGET" "$@"
fi
